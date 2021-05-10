import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
from NS_compute_block import *
from NS_transformer import TransformerLayers

strategy = tf.distribute.MirroredStrategy()

class NSModelDataOnly(keras.Model):
  def __init__(self, global_batch_size=64, alpha=[1.0, 1.0, 1.0], saveGradStat=False, **kwargs):
    super(NSModelDataOnly, self).__init__(**kwargs)
    self.alpha = alpha

    # ---- dicts for metrics and statistics ---- #
    # save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    self.global_batch_size = global_batch_size
    # create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    for key in ['loss', 'uMse', 'vMse', 'pMse','nuMse']:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    self.trainMetrics['rMse'] = keras.metrics.Mean(name='train_rMse')
    ## add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i in range(len(width)):
        for prefix in ['u_', 'v_', 'p_','nu_']:
          for suffix in ['w_avg', 'w_std', 'b_avg', 'b_std']:
            key = prefix + repr(i) + suffix
            self.trainMetrics[key] = keras.metrics.Mean(name='train '+key)
    # statistics
    self.trainStat = {}
    self.validStat = {}


  def call(self, inputs):
    return inputs


  def train_step(self, data):
    inputs = data[0]
    true    = data[1]
    singlesample=tf.shape(true)[1]
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    with tf.GradientTape(persistent=True) as tape0:

      flowPred = self(inputs)
      uMse    = mse(flowPred[:,:,:,0],true[:,:,:,0])
      uMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
      uMseGlobal = tf.nn.compute_average_loss(uMse, global_batch_size = self.global_batch_size)

      vMse    = mse(flowPred[:,:,:,1],true[:,:,:,1])
      vMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
      vMseGlobal = tf.nn.compute_average_loss(vMse, global_batch_size=self.global_batch_size)
      
      pMse    = mse(flowPred[:,:,:,2],true[:,:,:,2])
      pMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
      pMseGlobal = tf.nn.compute_average_loss(pMse, global_batch_size=self.global_batch_size)
      
      
      nuMse    = mse(flowPred[:,:,:,3],true[:,:,:,3])
      nuMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
      nuMseGlobal = tf.nn.compute_average_loss(nuMse, global_batch_size=self.global_batch_size)
      
      rMse    = tf.add_n(self.losses)
   
      loss    = 0.25*(uMseGlobal + vMseGlobal + pMseGlobal + nuMseGlobal) 
    # update gradients and trainable variables
    if self.saveGradStat:
      uMseGrad    = tape0.gradient(uMse, self.trainable_variables)
      vMseGrad    = tape0.gradient(vMse, self.trainable_variables)
      pMseGrad    = tape0.gradient(pMse, self.trainable_variables)
      nuMseGrad    = tape0.gradient(nuMse, self.trainable_variables)
    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['uMse'].update_state(uMseGlobal)
    self.trainMetrics['vMse'].update_state(vMseGlobal)
    self.trainMetrics['pMse'].update_state(pMseGlobal)
    self.trainMetrics['nuMse'].update_state(nuMseGlobal)
    self.trainMetrics['rMse'].update_state(rMse)
    # track gradients coefficients
    if self.saveGradStat:
      self.record_layer_gradient(uMseGrad, 'u_')
      self.record_layer_gradient(vMseGrad, 'v_')
      self.record_layer_gradient(pMseGrad, 'p_')
      self.record_layer_gradient(nuMseGrad, 'nu_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()

    return self.trainStat


  def test_step(self, data):
    inputs = data[0]
    true    = data[1]
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    s = tf.shape(true)[1]
    flowPred = self(inputs)
    uMse    = mse(flowPred[:,:,:,0],true[:,:,:,0])
    uMse    /= tf.cast(tf.reduce_prod(s), tf.float32)
    uMseGlobal = tf.nn.compute_average_loss(uMse, global_batch_size = self.global_batch_size)
    
    vMse    = mse(flowPred[:,:,:,1],true[:,:,:,1])
    vMse    /= tf.cast(tf.reduce_prod(s), tf.float32)
    vMseGlobal = tf.nn.compute_average_loss(vMse, global_batch_size=self.global_batch_size)
    
    pMse    = mse(flowPred[:,:,:,2],true[:,:,:,2])
    pMse    /= tf.cast(tf.reduce_prod(s), tf.float32)
    pMseGlobal = tf.nn.compute_average_loss(pMse, global_batch_size=self.global_batch_size)
    
    nuMse    = mse(flowPred[:,:,:,3],true[:,:,:,3])
    nuMse    /= tf.cast(tf.reduce_prod(s), tf.float32)
    nuMseGlobal = tf.nn.compute_average_loss(nuMse, global_batch_size=self.global_batch_size)
    
   
    loss    = 0.25*(uMseGlobal + vMseGlobal + pMseGlobal + nuMseGlobal) 

    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMseGlobal)
    self.validMetrics['vMse'].update_state(vMseGlobal)
    self.validMetrics['pMse'].update_state(pMseGlobal)
    self.validMetrics['nuMse'].update_state(nuMseGlobal)
    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    #
    return self.validStat


  def reset_metrics(self):
    for key in self.trainMetrics:
      self.trainMetrics[key].reset_states()
    for key in self.validMetrics:
      self.validMetrics[key].reset_states()


  def summary(self):
    nVar = 0
    for t in self.trainable_variables:
      print(t.name, t.shape)
      nVar += tf.reduce_prod(t.shape)
    print('{} trainable variables'.format(nVar))


  def record_layer_gradient(self, grads, baseName):
    '''
    record the average and standard deviation of each layer's
    weights and biases
    '''
    for i, g in enumerate(grads):
      if g != None:
        l = i // 2
        parameter = 'w' if i%2==0 else 'b'
        prefix = baseName + '_{:d}{}_'.format(l,parameter)
        gAbs = tf.abs(g)
        gAvg = tf.reduce_mean(gAbs)
        gStd = tf.reduce_mean(tf.square(gAbs - gAvg))
        self.trainMetrics[prefix+'avg'].update_state(gAvg)
        self.trainMetrics[prefix+'std'].update_state(gStd)


  def gradient_bc(self, bc, xy):
    '''
    bc - 1D array of all boundary values (u, v, p)
    xy - [batch, 2], including all points to infer
    '''
    xy      = tf.convert_to_tensor(xy)
    # bc      = tf.convert_to_tensor(bc)
    bcTiled = tf.convert_to_tensor(bc)
    bcTiled = tf.expand_dims(bcTiled, 0)
    bcTiled = tf.tile(bcTiled, (xy.shape[0], 1))

    with tf.GradientTape(watch_accessed_variables=False,\
                         persistent=True) as tape:
      tape.watch(bcTiled)
      uvp = self([bcTiled, xy])
      u, v, p = uvp[:,0], uvp[:,1], uvp[:,2]
    u_bc = tape.gradient(u, bcTiled)
    v_bc = tape.gradient(v, bcTiled)
    p_bc = tape.gradient(p, bcTiled)
    del tape
    uvp_bc = tf.stack([u_bc, v_bc, p_bc], 1)
    return uvp_bc.numpy()

class NSModelSymmCNN(NSModelDataOnly):

  def __init__(self, 
               input_shape=(64,256,4),
               filters=[4,16,32,256],
               kernel_size=(5,5),
	       activation="LeakyReLU",
               strides=(1,1), 
               reg=None, 
               lastLinear = False, 
               **kwargs):

    super(NSModelSymmCNN, self).__init__(**kwargs)
    self.convdeconv= ConvolutionDeconvolutionLayers(input_shape=input_shape,
							filters=filters,
							kernel_size=kernel_size,
							strides=strides,
							activation=activation)

  def call(self, inputs, training=True):
    return self.convdeconv(inputs)



class NSModelMLP(NSModelDataOnly):
  '''
  Feed-Forward Model takes in the coordinates and variables on boundary,
  Re, and the collocation point as input, outputs (u, v, p)
  '''
  def __init__(self, width=[256, 256, 256, 128, 128, 128, 64, 32, 3],\
               reg=None, lastLinear = False, **kwargs):
    super(NSModelMLP, self).__init__(**kwargs)
    self.width = width
    self.reg   = reg
    if reg == None:
      self.mlp = DenseLayers(width=width, prefix='bc', \
                             last_linear=lastLinear)
    else:
      self.mlp = DenseLayers(width=width, reg=reg, prefix='bc', \
                             last_linear=lastLinear)

  def call(self, inputs):
    '''
    inputs: [bcXybcRe, xyColloc]
    '''
    bcXybcReXy = tf.concat(inputs, axis=-1)
    return self.mlp(bcXybcReXy)


  def preview(self):
    print('--------------------------------')
    print('model preview')
    print('--------------------------------')
    print('fully connected network:')
    print(self.width)
    print('layer regularization')
    print(self.reg)
    print('--------------------------------')


class NSModelMlpRes(NSModelDataOnly):
  '''
  Feed-Forward Model takes in the coordinates and variables on boundary,
  Re, and the collocation point as input, outputs (u, v, p)
  '''
  def __init__(self, resWidth=[256, 128],\
               widthAfterRes=[64, 32, 3], **kwargs):
    super(NSModelMlpRes, self).__init__(**kwargs)
    assert len(widthAfterRes) > 0  and len(resWidth) > 0
    self.mlp   = DenseLayers(width=widthAfterRes, prefix='after')
    self.resLayers = []
    for w in resWidth:
      self.resLayers.append(DenseResidualLayers(width=w))

  def call(self, inputs):
    '''
    inputs: [bcXybcRe, xyColloc]
    '''
    uvp = tf.concat(inputs, axis=-1)
    for l in self.resLayers:
      uvp = l(uvp)
    uvp = self.mlp(uvp)
    return uvp



#---------------------------------------------------------------
# PINN model
#---------------------------------------------------------------
class NSModelPinn(keras.Model):
  def __init__(self, 
               inputshape = [64,256,4],
               filters=[4,32,128,256],\
               kernel_size = (5,5),
               strides = (1,1),
               beta = [1.0, 1.0, 1.0], \
               global_batch_size=64,
               activation="LeakyReLU",
               reg=None, 
               saveGradStat=False, 
               **kwargs):

    super(NSModelPinn, self).__init__(**kwargs)
    self.inputshape = inputshape
    self.convdeconv= ConvolutionDeconvolutionLayers(input_shape=self.inputshape,
							filters=filters,
							kernel_size=kernel_size, 		
							strides=strides,
							activation=activation)
      
    # coefficient for data and pde loss
    self.beta  = beta

    # ---- dicts for metrics and statistics ---- #
    # save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    names = ['loss', 'data_loss','cont_loss','mom_x_loss','mom_z_loss', 'uMse', 'vMse', 'pMse','nuMse']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    ## add metrics for layers' weights, if save_grad_stat is required
    ## i even for weights, odd for bias
    if self.saveGradStat:
      for i in range(len(width)):
        for prefix in ['u_', 'v_', 'p_', 'pde0_', 'pde1_', 'pde2_']:
          for suffix in ['w_avg', 'w_std', 'b_avg', 'b_std']:
            key = prefix + repr(i) + suffix
            self.trainMetrics[key] = keras.metrics.Mean(name='train '+key)
    # statistics
    self.trainStat = {}
    self.validStat = {}
    self.global_batch_size = global_batch_size

  def call(self, inputs, training=True):
   
   toCNN = tf.concat([inputs[0],inputs[1]],axis=-1)
   return self.convdeconv(toCNN)


  def record_layer_gradient(self, grads, baseName):
    '''
    record the average and standard deviation of each layer's
    weights and biases
    '''
    for i, g in enumerate(grads):
      if g != None:
        l = i // 2
        parameter = 'w' if i%2==0 else 'b'
        prefix = baseName + '_{:d}{}_'.format(l,parameter)
        gAbs = tf.abs(g)
        gAvg = tf.reduce_mean(gAbs)
        gStd = tf.reduce_mean(tf.square(gAbs - gAvg))
        self.trainMetrics[prefix+'avg'].update_state(gAvg)
        self.trainMetrics[prefix+'std'].update_state(gStd)


  def compute_data_pde_losses(self, uvpnu_input,uvpnu_labels,xz):
    # track computation for 2nd derivatives for u, v, p
    
    
    singlesample=tf.shape(uvpnu_labels)[1]
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xz)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xz)
        flowPred = self([uvpnu_input,xz])
        u_pred       = flowPred[:,:,:,0]
        v_pred       = flowPred[:,:,:,1]
        p_pred       = flowPred[:,:,:,2]
        nu_pred       = flowPred[:,:,:,3]
      # 1st order derivatives
      u_grad   = tape1.gradient(u_pred, xz)
      v_grad   = tape1.gradient(v_pred, xz)
      p_grad   = tape1.gradient(p_pred, xz)
      u_x, u_z = u_grad[:,:,:,0], u_grad[:,:,:,1]
      v_x, v_z = v_grad[:,:,:,0], v_grad[:,:,:,1]
      p_x, p_z = p_grad[:,:,:,0], p_grad[:,:,:,1]
      del tape1
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xz)[:,:,:,0]
    u_zz = tape2.gradient(u_z, xz)[:,:,:,1]
    v_xx = tape2.gradient(v_x, xz)[:,:,:,0]
    v_zz = tape2.gradient(v_z, xz)[:,:,:,1]
    del tape2

    # compute data loss
    uMse    = mse(u_pred,uvpnu_labels[:,:,:,0])
    uMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    uMseGlobal = tf.nn.compute_average_loss(uMse, global_batch_size = self.global_batch_size)

    vMse    = mse(v_pred,uvpnu_labels[:,:,:,1])
    vMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    vMseGlobal = tf.nn.compute_average_loss(vMse, global_batch_size=self.global_batch_size)
      
    pMse    = mse(p_pred,uvpnu_labels[:,:,:,2])
    pMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    pMseGlobal = tf.nn.compute_average_loss(pMse, global_batch_size=self.global_batch_size)
    
    
    nuMse    = mse(nu_pred,uvpnu_labels[:,:,:,3])
    nuMse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    nuMseGlobal = tf.nn.compute_average_loss(nuMse, global_batch_size=self.global_batch_size)

    # pde error, 0 continuity, 1-2 NS
    pde0    = u_x + v_z
    z = tf.zeros(tf.shape(pde0),dtype=tf.float32)
    pde0Mse    = mse(pde0,z)
    pde0Mse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    pde0MseGlobal = tf.nn.compute_average_loss(pde0Mse, global_batch_size=self.global_batch_size)

    #viscosity = tf.constant(0.01,shape=tf.shape(pde0),dtype=tf.float32) 
    #invRe = tf.constant(1/6000,shape=tf.shape(pde0),dtype=tf.float32) 

    pde1    = u_pred*u_x + v_pred*u_z + p_x - (0.01+ nu_pred)*(1/6000)*(u_xx + u_zz)
    pde1Mse    = mse(pde1,z)
    pde1Mse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    pde1MseGlobal = tf.nn.compute_average_loss(pde1Mse, global_batch_size=self.global_batch_size)

    pde2    = u_pred*v_x + v_pred*v_z + p_z - (0.01 + nu_pred)*(1/6000)*(v_xx + v_zz)
    pde2Mse    = mse(pde2,z)
    pde2Mse    /= tf.cast(tf.reduce_prod(singlesample), tf.float32)
    pde2MseGlobal = tf.nn.compute_average_loss(pde2Mse, global_batch_size=self.global_batch_size)

    return uMseGlobal, vMseGlobal, pMseGlobal, nuMseGlobal, pde0MseGlobal, pde1MseGlobal, pde2MseGlobal


  def train_step(self, data):

    inputs = data[0]
    labels = data[1]

    uvpnu_input = inputs[:,:,:,0:4]
    xz          = inputs[:,:,:,4:6]
    uvpnu_labels = labels[:,:,:,0:4]

    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(uvpnu_input,uvpnu_labels,xz)
      # replica's loss, divided by global batch size
      data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

      loss = data_loss + self.beta[0]*contMse + self.beta[1]*momxMse + self.beta[2]*momzMse

      loss += tf.add_n(self.losses)
    # update gradients
    if self.saveGradStat:
      uMseGrad    = tape0.gradient(uMse,    self.trainable_variables)
      vMseGrad    = tape0.gradient(vMse,    self.trainable_variables)
      pMseGrad    = tape0.gradient(pMse,    self.trainable_variables)
      pdeMse0Grad = tape0.gradient(pdeMse0, self.trainable_variables)
      pdeMse1Grad = tape0.gradient(pdeMse1, self.trainable_variables)
      pdeMse2Grad = tape0.gradient(pdeMse2, self.trainable_variables)
    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['data_loss'].update_state(data_loss)
    self.trainMetrics['cont_loss'].update_state(contMse)
    self.trainMetrics['mom_x_loss'].update_state(momxMse)
    self.trainMetrics['mom_z_loss'].update_state(momzMse)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    self.trainMetrics['nuMse'].update_state(nuMse)
    # track gradients coefficients
    if self.saveGradStat:
      self.record_layer_gradient(uMseGrad, 'u_')
      self.record_layer_gradient(vMseGrad, 'v_')
      self.record_layer_gradient(pMseGrad, 'p_')
      self.record_layer_gradient(pdeMse0Grad, 'pde0_')
      self.record_layer_gradient(pdeMse1Grad, 'pde1_')
      self.record_layer_gradient(pdeMse2Grad, 'pde2_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):

    inputs = data[0]
    labels = data[1]

    uvpnu_input = inputs[:,:,:,0:4]
    xz          = inputs[:,:,:,4:6]
    uvpnu_labels = labels[:,:,:,0:4]

    uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(uvpnu_input,uvpnu_labels,xz)
      # replica's loss, divided by global batch size
    data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

    loss = data_loss + self.beta[0]*contMse + self.beta[1]*momxMse + self.beta[2]*momzMse

    loss += tf.add_n(self.losses)
    # track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(contMse)
    self.validMetrics['mom_x_loss'].update_state(momxMse)
    self.validMetrics['mom_z_loss'].update_state(momzMse)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['nuMse'].update_state(nuMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


  def reset_metrics(self):
    for key in self.trainMetrics:
      self.trainMetrics[key].reset_states()
    for key in self.validMetrics:
      self.validMetrics[key].reset_states()


  def summary(self):
    nVar = 0
    for t in self.trainable_variables:
      print(t.name, t.shape)
      nVar += tf.reduce_prod(t.shape)
    print('{} trainalbe variables'.format(nVar))


  def preview(self):
    print('--------------------------------')
    print('model preview')
    print('--------------------------------')
    print('fully connected network:')
    print(self.width)
    print('layer regularization')
    print(self.reg)
    print('--------------------------------')


def space_gradient(nn, bc, xy):
  # input bc should be a 1d tensor, expand it to 2D
  # to match xy, which is a 2D tensor [batch, 2]
  bc = tf.expand_dims(bc, 0)
  bc = tf.tile(bc, (xy.shape[0], 1))
  with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(xy)
    tape2.watch(bc)
    with tf.GradientTape(persistent=True) as tape1:
      tape1.watch(xy)
      tape1.watch(bc)
      uvp = nn([bc, xy])
      u, v, p = uvp[:,0], uvp[:,1], uvp[:,2]
    u_1 = tape1.gradient(u, xy)
    v_1 = tape1.gradient(v, xy)
    del tape1
    u_x, u_y = u_1[:,0], u_1[:,1]
    v_x, v_y = v_1[:,0], v_1[:,1]
  u_xx = tape2.gradient(u_x, xy)[:,0]
  u_yy = tape2.gradient(u_y, xy)[:,1]
  v_xx = tape2.gradient(v_x, xy)[:,0]
  v_yy = tape2.gradient(v_y, xy)[:,1]
  del tape2
  u_2  = tf.stack([u_xx, u_yy], axis=-1)
  v_2  = tf.stack([v_xx, v_yy], axis=-1)

  return [u_1, u_2, v_1, v_2]


def infer_range(nn, bottom, right, top, left, xy):
  # concatenate four edges
  bc = [bottom, right, tf.reverse(top, [0]), tf.reverse(left, [0])]
  bc = tf.concat(bc, 0)
  # flatten bc as input
  bc  = tf.reshape(bc, [-1])
  # tile bc according to input xy
  shape3d = xy.shape
  xy = tf.reshape(xy, (shape3d[0]*shape3d[1], 2))
  bc = tf.expand_dims(bc, 0)
  bc = tf.tile(bc, (shape3d[0]*shape3d[1], 1))
  # infer
  uvp = nn([bc, xy])
  uvp  = tf.reshape(uvp, (shape3d[0], shape3d[1], 3))
  return uvp



class NSModelTransformerPinn(NSModelPinn):

     
  def __init__(self, 
               image_size=[64,256,6],
               patch_size=[32,128],
               projection_dim_encoder=768
	       projection_dim_attention=64,
               num_heads=4, 
               transformer_layers=1,
               **kwargs):

    super(NSModelTransformerPinn, self).__init__(**kwargs)
    
    self.nPatchesImage = (image_size[0]*image_size[1] // (patch_size[0]*patch_size[1]) )
    self.nRowsPatch = patch_size[0]

    self.patch_size = patch_size

    self.transformer = VisionTransformerLayers(image_size=image_size,
				         patch_size = self.patch_size,
                                         projection_dim_encoder = projection_dim_encoder,
                                         projection_dim_attention = projection_dim_attention,
                                         num_heads = num_heads,
                                         transformer_layers=transformer_layers,
                                         )

  def call(self, inputs, training=True):
    to_transformer = tf.concat([inputs[0],inputs[1]],axis=-1)
    return self.transformer(to_transformer)



  def initialize(self,model_name='ViT-B_16.npz')

   weights = self.transformer.layers[5].get_weights()
   
   return weights


  def compute_data_pde_losses(self, uvpnu_input,uvpnu_labels,xz):

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xz)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xz)
        flowPred = self([uvpnu_input,xz])
        u_pred       = flowPred[:,:,:,:,0]
        v_pred       = flowPred[:,:,:,:,1]
        p_pred       = flowPred[:,:,:,:,2]
        nu_pred       = flowPred[:,:,:,:,3]

      # 1st order derivatives
      u_grad   = tape1.gradient(u_pred, xz)
      v_grad   = tape1.gradient(v_pred, xz)
      p_grad   = tape1.gradient(p_pred, xz)
      u_x, u_z = u_grad[:,:,:,:,0], u_grad[:,:,:,:,1]
      v_x, v_z = v_grad[:,:,:,:,0], v_grad[:,:,:,:,1]
      p_x, p_z = p_grad[:,:,:,:,0], p_grad[:,:,:,:,1]
      del tape1
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xz)[:,:,:,:,0]
    u_zz = tape2.gradient(u_z, xz)[:,:,:,:,1]
    v_xx = tape2.gradient(v_x, xz)[:,:,:,:,0]
    v_zz = tape2.gradient(v_z, xz)[:,:,:,:,1]
    del tape2


    uMse = mse(u_pred,uvpnu_labels[:,:,:,:,0]) 
    uMseGlobal = tf.nn.compute_average_loss(uMse, global_batch_size = self.global_batch_size*self.nPatchesImage*self.nRowsPatch)

    vMse    = mse(v_pred,uvpnu_labels[:,:,:,:,1])
    vMseGlobal = tf.nn.compute_average_loss(vMse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)
      
    pMse    = mse(p_pred,uvpnu_labels[:,:,:,:,2])
    pMseGlobal = tf.nn.compute_average_loss(pMse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)
    
    nuMse    = mse(nu_pred,uvpnu_labels[:,:,:,:,3])
    nuMseGlobal = tf.nn.compute_average_loss(nuMse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)

    # pde error, 0 continuity, 1-2 NS
    pde0    = u_x + v_z
    z = tf.zeros(tf.shape(pde0),dtype=tf.float32)
    pde0Mse    = mse(pde0,z)
    pde0MseGlobal = tf.nn.compute_average_loss(pde0Mse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)

    pde1    = u_pred*u_x + v_pred*u_z + p_x - (0.01+ nu_pred)*(1/6000)*(u_xx + u_zz)
    pde1Mse    = mse(pde1,z)
    pde1MseGlobal = tf.nn.compute_average_loss(pde1Mse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)

    pde2    = u_pred*v_x + v_pred*v_z + p_z - (0.01 + nu_pred)*(1/6000)*(v_xx + v_zz)
    pde2Mse    = mse(pde2,z)
    pde2MseGlobal = tf.nn.compute_average_loss(pde2Mse, global_batch_size=self.global_batch_size*self.nPatchesImage*self.nRowsPatch)

    return uMseGlobal, vMseGlobal, pMseGlobal, nuMseGlobal, pde0MseGlobal, pde1MseGlobal, pde2MseGlobal

  def train_step(self, data):

    inputs = data[0]
    labels = data[1]

    uvpnu_input = inputs[:,:,:,:,0:4]
    xz          = inputs[:,:,:,:,4:6]
    uvpnu_labels = labels[:,:,:,:,0:4]

    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(uvpnu_input,uvpnu_labels,xz)
      # replica's loss, divided by global batch size
      data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

      loss = data_loss + self.beta[0]*contMse + self.beta[1]*momxMse + self.beta[2]*momzMse

#      loss += tf.add_n(self.losses)
    # update gradients
    if self.saveGradStat:
      uMseGrad    = tape0.gradient(uMse,    self.trainable_variables)
      vMseGrad    = tape0.gradient(vMse,    self.trainable_variables)
      pMseGrad    = tape0.gradient(pMse,    self.trainable_variables)
      pdeMse0Grad = tape0.gradient(pdeMse0, self.trainable_variables)
      pdeMse1Grad = tape0.gradient(pdeMse1, self.trainable_variables)
      pdeMse2Grad = tape0.gradient(pdeMse2, self.trainable_variables)
    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['data_loss'].update_state(data_loss)
    self.trainMetrics['cont_loss'].update_state(contMse)
    self.trainMetrics['mom_x_loss'].update_state(momxMse)
    self.trainMetrics['mom_z_loss'].update_state(momzMse)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    self.trainMetrics['nuMse'].update_state(nuMse)
    # track gradients coefficients
    if self.saveGradStat:
      self.record_layer_gradient(uMseGrad, 'u_')
      self.record_layer_gradient(vMseGrad, 'v_')
      self.record_layer_gradient(pMseGrad, 'p_')
      self.record_layer_gradient(pdeMse0Grad, 'pde0_')
      self.record_layer_gradient(pdeMse1Grad, 'pde1_')
      self.record_layer_gradient(pdeMse2Grad, 'pde2_')
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat


  def test_step(self, data):

    inputs = data[0]
    labels = data[1]

    uvpnu_input = inputs[:,:,:,:,0:4]
    xz          = inputs[:,:,:,:,4:6]
    uvpnu_labels = labels[:,:,:,:,0:4]

    uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(uvpnu_input,uvpnu_labels,xz)
      # replica's loss, divided by global batch size
    data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

    loss = data_loss + self.beta[0]*contMse + self.beta[1]*momxMse + self.beta[2]*momzMse

    #loss += tf.add_n(self.losses)
    # track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(contMse)
    self.validMetrics['mom_x_loss'].update_state(momxMse)
    self.validMetrics['mom_z_loss'].update_state(momzMse)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['nuMse'].update_state(nuMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat

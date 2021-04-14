import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
from NS_compute_block import *

strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE=1

class NSModelDataOnly(keras.Model):
  def __init__(self, alpha=[1.0, 1.0, 1.0], saveGradStat=False, **kwargs):
    super(NSModelDataOnly, self).__init__(**kwargs)
    self.alpha = alpha

    # ---- dicts for metrics and statistics ---- #
    # save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    for key in ['loss', 'uMse', 'vMse', 'pMse','nuMse']:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)
    self.trainMetrics['rMse'] = keras.metrics.Mean(name='train_rMse')
    for key in ['uMae', 'vMae', 'pMae','nuMae']:
      self.trainMetrics[key] = keras.metrics.MeanAbsoluteError(name='train_'+key)
      self.validMetrics[key] = keras.metrics.MeanAbsoluteError(name='valid_'+key)
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
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    with tf.GradientTape(persistent=True) as tape0:

      flowPred = self(inputs)
      uMse    = mse(flowPred[:,:,:,0],true[:,:,:,0])
      uMse    /= tf.cast(tf.reduce_prod(tf.shape(true)[0:-1]), tf.float32)
      uMseGlobal = tf.nn.compute_average_loss(uMse, global_batch_size = GLOBAL_BATCH_SIZE)
      
      vMse    = mse(flowPred[:,:,:,1],true[:,:,:,1])
      vMse    /= tf.cast(tf.reduce_prod(tf.shape(true)[0:-1]), tf.float32)
      vMseGlobal = tf.nn.compute_average_loss(vMse, global_batch_size=GLOBAL_BATCH_SIZE)
      
      pMse    = mse(flowPred[:,:,:,2],true[:,:,:,2])
      pMse    /= tf.cast(tf.reduce_prod(tf.shape(true)[0:-1]), tf.float32)
      pMseGlobal = tf.nn.compute_average_loss(pMse, global_batch_size=GLOBAL_BATCH_SIZE)
      
      
      nuMse    = mse(flowPred[:,:,:,3],true[:,:,:,3])
      nuMse    /= tf.cast(tf.reduce_prod(tf.shape(true)[0:-1]), tf.float32)
      nuMseGlobal = tf.nn.compute_average_loss(nuMse, global_batch_size=GLOBAL_BATCH_SIZE)
      
      rMse    = tf.add_n(self.losses)
   
      loss    = 0.25*(uMse + vMse + pMse + nuMse) 
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
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    self.trainMetrics['nuMse'].update_state(pMse)
    self.trainMetrics['rMse'].update_state(rMse)
    self.trainMetrics['uMae'].update_state(flowPred[:,0], true[:,0])
    self.trainMetrics['vMae'].update_state(flowPred[:,1], true[:,1])
    self.trainMetrics['pMae'].update_state(flowPred[:,2], true[:,2])
    self.trainMetrics['nuMae'].update_state(flowPred[:,3], true[:,3])
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
    inputs  = data[0]
    uvp     = data[1]
    # predict
    uvpPred = self(inputs)
    # update loss
    uMse    = tf.reduce_mean(tf.square(uvpPred[:,:,:,0] - uvp[:,:,:,0]))
    vMse    = tf.reduce_mean(tf.square(uvpPred[:,:,:,1] - uvp[:,:,:,1]))
    pMse    = tf.reduce_mean(tf.square(uvpPred[:,:,:,2] - uvp[:,:,:,2]))
    nuMse    = tf.reduce_mean(tf.square(uvpPred[:,:,:,3] - uvp[:,:,:,3]))
    loss    = 0.25*(uMse +vMse +pMse+ nuMse)
    # track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['nuMse'].update_state(nuMse)
    self.validMetrics['uMae'].update_state(uvpPred[:,0], uvp[:,0])
    self.validMetrics['vMae'].update_state(uvpPred[:,1], uvp[:,1])
    self.validMetrics['pMae'].update_state(uvpPred[:,2], uvp[:,2])
    self.validMetrics['nuMae'].update_state(uvpPred[:,3], uvp[:,3])
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
               strides=(1,1), 
               reg=None, 
               lastLinear = False, 
               **kwargs):

    super(NSModelSymmCNN, self).__init__(**kwargs)
    self.convdeconv= ConvolutionDeconvolutionLayers(input_shape=input_shape,filters=filters,kernel_size=kernel_size,strides=strides)

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
    print('coefficients for data loss {} {} {}'.format(\
          self.alpha[0], self.alpha[1], self.alpha[2]))
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
  def __init__(self, width=[256, 256, 256, 128, 128, 128, 64, 32, 3],\
               alpha = [1.0, 1.0, 1.0], beta = [1e-4, 1e-4, 1e-4], \
               reg=None, saveGradStat=False, **kwargs):
    super(NSModelPinn, self).__init__(**kwargs)
    self.width = width
    self.reg   = reg
    if reg == None:
      self.mlp = DenseLayers(width=width, prefix='bc')
    else:
      self.mlp = DenseLayers(width=width, reg=reg, prefix='bc')
      
    # coefficient for data and pde loss
    self.alpha = alpha
    self.beta  = beta

    # ---- dicts for metrics and statistics ---- #
    # save gradients' statistics per layer
    self.saveGradStat = saveGradStat
    # create metrics
    self.trainMetrics = {}
    self.validMetrics = {}
    # add metrics
    names = ['loss', 'uMse', 'vMse', 'pMse', 'pde0', 'pde1', 'pde2', \
             'uMae', 'vMae', 'pMae']
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


  def call(self, inputs):
    '''
    inputs: [bc, xy, w]
    '''
    bcXy = tf.concat([inputs[0], inputs[1]], axis=-1)
    return self.mlp(bcXy)


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


  def compute_data_pde_losses(self, bc, xy, w, uvp):
    # track computation for 2nd derivatives for u, v, p
    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(xy)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(xy)
        uvpPred = self([bc, xy, w])
        u       = uvpPred[:,0]
        v       = uvpPred[:,1]
        p       = uvpPred[:,2]
      # 1st order derivatives
      u_grad   = tape1.gradient(u, xy)
      v_grad   = tape1.gradient(v, xy)
      p_grad   = tape1.gradient(p, xy)
      u_x, u_y = u_grad[:,0], u_grad[:,1]
      v_x, v_y = v_grad[:,0], v_grad[:,1]
      p_x, p_y = p_grad[:,0], p_grad[:,1]
      del tape1
    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, xy)[:,0]
    u_yy = tape2.gradient(u_y, xy)[:,1]
    v_xx = tape2.gradient(v_x, xy)[:,0]
    v_yy = tape2.gradient(v_y, xy)[:,1]
    del tape2

    # compute data loss
    w = tf.squeeze(w)
    nDataPoint = tf.reduce_sum(w) + 1.0e-10
    uMse  = tf.reduce_sum(w * tf.square(u - uvp[:,0])) / nDataPoint
    vMse  = tf.reduce_sum(w * tf.square(v - uvp[:,1])) / nDataPoint
    pMse  = tf.reduce_sum(w * tf.square(p - uvp[:,2])) / nDataPoint
    # pde error, 0 continuity, 1-2 NS
    ww      = 1.0 - w
    nPdePoint = tf.reduce_sum(ww) + 1.0e-10
    pde0    = u_x + v_y
    pde1    = u*u_x + v*u_y + p_x - (u_xx + u_yy)/500.0
    pde2    = u*v_x + v*v_y + p_y - (v_xx + v_yy)/500.0
    pdeMse0 = tf.reduce_sum(tf.square(pde0) * ww) / nPdePoint
    pdeMse1 = tf.reduce_sum(tf.square(pde1) * ww) / nPdePoint
    pdeMse2 = tf.reduce_sum(tf.square(pde2) * ww) / nPdePoint

    return uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2


  def train_step(self, data):
    bc  = data[0][0]
    xy  = data[0][1]
    w   = data[0][2]
    uvp = data[1]
    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2 = \
        self.compute_data_pde_losses(bc, xy, w, uvp)
      # replica's loss, divided by global batch size
      loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse   + self.alpha[2]*pMse \
              + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2)
      loss += tf.add_n(self.losses)
      loss  = loss / strategy.num_replicas_in_sync
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
    self.trainMetrics['loss'].update_state(loss*strategy.num_replicas_in_sync)
    self.trainMetrics['uMse'].update_state(uMse)
    self.trainMetrics['vMse'].update_state(vMse)
    self.trainMetrics['pMse'].update_state(pMse)
    self.trainMetrics['pde0'].update_state(pdeMse0)
    self.trainMetrics['pde1'].update_state(pdeMse1)
    self.trainMetrics['pde2'].update_state(pdeMse2)
    w = tf.squeeze(w)
    nDataPoint = tf.reduce_sum(w) + 1.0e-10
    uMae = tf.reduce_sum(tf.abs((uvpPred[:,0] - uvp[:,0]) * w))/nDataPoint
    vMae = tf.reduce_sum(tf.abs((uvpPred[:,1] - uvp[:,1]) * w))/nDataPoint
    pMae = tf.reduce_sum(tf.abs((uvpPred[:,2] - uvp[:,2]) * w))/nDataPoint
    self.trainMetrics['uMae'].update_state(uMae)
    self.trainMetrics['vMae'].update_state(vMae)
    self.trainMetrics['pMae'].update_state(pMae)
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
    bc  = data[0][0]
    xy  = data[0][1]
    w   = data[0][2]
    uvp = data[1]

    # compuate the data and pde losses
    uvpPred, uMse, vMse, pMse, pdeMse0, pdeMse1, pdeMse2 = \
      self.compute_data_pde_losses(bc, xy, w, uvp)
    # replica's loss, divided by global batch size
    loss  = ( self.alpha[0]*uMse   + self.alpha[1]*vMse   + self.alpha[2]*pMse \
            + self.beta[0]*pdeMse0 + self.beta[1]*pdeMse1 + self.beta[2]*pdeMse2)

    # track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['uMse'].update_state(uMse)
    self.validMetrics['vMse'].update_state(vMse)
    self.validMetrics['pMse'].update_state(pMse)
    self.validMetrics['pde0'].update_state(pdeMse0)
    self.validMetrics['pde1'].update_state(pdeMse1)
    self.validMetrics['pde2'].update_state(pdeMse2)
    w = tf.squeeze(w)
    nDataPoint = tf.reduce_sum(w) + 1.0e-10
    uMae = tf.reduce_sum(tf.abs((uvpPred[:,0] - uvp[:,0]) * w))/nDataPoint
    vMae = tf.reduce_sum(tf.abs((uvpPred[:,1] - uvp[:,1]) * w))/nDataPoint
    pMae = tf.reduce_sum(tf.abs((uvpPred[:,2] - uvp[:,2]) * w))/nDataPoint
    self.validMetrics['uMae'].update_state(uMae)
    self.validMetrics['vMae'].update_state(vMae)
    self.validMetrics['pMae'].update_state(pMae)

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
    print('coefficients for data loss {} {} {}'.format(\
          self.alpha[0], self.alpha[1], self.alpha[2]))
    print('coefficients for pde residual {} {} {}'.format(\
          self.beta[0], self.beta[1], self.beta[2]))
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
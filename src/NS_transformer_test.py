import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn


class Projection(keras.layers.Layer):
    def __init__(self,projection_dim_encoder,patch_size):
     super(Projection,self).__init__()
     self.projection = keras.layers.Conv2D(filters=projection_dim_encoder,
                                           kernel_size=patch_size,
                                           strides=patch_size,
                                           padding="valid",
                                           name="Pre/Projection",
                                           trainable=False)
    def call(self,inputs):

      projection = self.projection(inputs)

      return projection

class PositionEmbedding(keras.layers.Layer):
    def __init__(self,sequence_length,projection_dim_encoder):
     super(PositionEmbedding,self).__init__()
     self.sequence_length = sequence_length
     self.position_embedding = keras.layers.Embedding(input_dim=sequence_length,
                                                      output_dim=projection_dim_encoder,
                                                      trainable=False,
                                                      name="Pre/PositionEmbedding")
    def call(self,inputs):
     positions = tf.range(start=0,limit=self.sequence_length,delta=1)
     embedding = self.position_embedding(positions)
     return inputs + embedding


#class Masking

class NSTransformer(NSModelPinn):
  def __init__(self, 
               image_size=[64,256,6],
               filter_size=[16,16],
               patch_size = [32,128],
               sequence_length = 196,
               projection_dim_encoder=768,
               projection_dim_attention=64,
               num_heads = 12,
               transformer_layers=12,
               **kwargs):

    super(NSTransformer, self).__init__(**kwargs)

    self.image_size = image_size
    self.patch_size = patch_size
    self.filter_size = filter_size
    self.sequence_length = sequence_length

    self.channelsInput = image_size[2]
    self.channelsOutput = 4

    self.nRowsImage = image_size[0]
    self.nColumnsImage = image_size[1]
    self.nPixelsImage = self.nRowsImage * self.nColumnsImage

    self.nRowsPatch = patch_size[0]
    self.nColumnsPatch = patch_size[1]
    self.nPixelsPatch = self.nRowsPatch * self.nColumnsPatch

    self.nPatchesImage = ( ( self.nPixelsImage  ) // (self.nPixelsPatch) )

    self.projection_dim_encoder = projection_dim_encoder
    self.projection_dim_attention = projection_dim_attention
    self.num_heads = num_heads
    self.transformer_layers = transformer_layers
    self.transformer_units = [
    self.projection_dim_encoder * 2,
    self.projection_dim_encoder,
                         ]
    self.Norm0 = []
    self.Norm1 = []
    self.Attention = []
    self.Dense0 = []
    self.Dense1 = []
    self.Dropout0 = []
    self.Dropout1 = []
    self.Add0 = []
    self.Add1 = []

    self.preprocess()
    self.encoder()
    self.task()

  def preprocess(self):

    self.InitialReshape = keras.layers.Reshape((self.image_size[0],self.image_size[1],self.image_size[2]))
    self.InitialDeconv =keras.layers.Conv2DTranspose(filters=3,kernel_size=(49,193),strides=(1,1),padding="valid",name="Pre/Deconv")
    self.Projection = Projection(self.projection_dim_encoder,self.filter_size)
    self.Reshape = keras.layers.Reshape((self.sequence_length,self.projection_dim_encoder))
    self.PositionEmbedding = PositionEmbedding(self.sequence_length,self.projection_dim_encoder)


  def task(self):

    self.Flatten = keras.layers.Flatten(name="Task/Flatten")
    self.MapDense = keras.layers.Dense(16,activation=tf.nn.gelu,name="Task/Dense")
    self.MapReshape0 = keras.layers.Reshape( (2,2,self.channelsOutput),name="Task/Reshape" )
    self.MapDeconv = keras.layers.Conv2DTranspose(filters=self.channelsOutput,kernel_size=(5,5),padding="same",strides=(32,128),activation='linear',name="Task/Conv")
    self.MapReshape1 = keras.layers.Reshape( (self.nPatchesImage, self.nRowsPatch, self.nColumnsPatch, self.channelsOutput),name="Task/Reshape_Out" )

  def encoder(self):

      for i in range(self.transformer_layers):
 
        name = 'T/EB_'+str(i)+'/'

        self.Norm0.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=False,name=name+'Norm0'))
        self.Attention.append(keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim_attention, dropout=0.1,trainable=False,name = name+'Attention'))
        self.Add0.append(keras.layers.Add(name=name+"Add0"))
        self.Norm1.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=False,name=name+"Norm1"))
        self.Dense0.append(keras.layers.Dense(4*self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False,name=name+"Dense0"))
        self.Dropout0.append(keras.layers.Dropout(rate=0.1,name=name+'Dropout0'))
        self.Dense1.append(keras.layers.Dense(self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False,name=name+"Dense1"))
        self.Dropout1.append(keras.layers.Dropout(rate=0.1,name=name+'Dropout1'))
        self.Add1.append(keras.layers.Add(name=name+"Add1"))



  def set_weights(self,vit):
     
    t = self.transformer_layers
    wvit = np.load(vit)
    projection = 9*t +2
    pos_embedding = 9*t +4

    for i in range(self.transformer_layers):
     norm0=i
     norm1=i+t
     attention = i+2*t
     dense0 = i+3*t
     dense1 = i+4*t
     dropout0 = i+5*t
     dropout1 = i+6*t
     add0 = i+7*t
     add1 =i +8*t

     w = []      
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias'])

     self.layers[norm0].set_weights(w)
    
     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias'])
  
     self.layers[norm1].set_weights(w)

     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias'])
     self.layers[attention].set_weights(w)
     
     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias'])
  
     self.layers[dense0].set_weights(w)

     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias'])
  
     self.layers[dense1].set_weights(w)

    w.clear()
    w.append(wvit['embedding/kernel'])
    w.append(wvit['embedding/bias'])
  
    self.layers[projection].set_weights(w)

    w.clear()
    w.append(wvit['Transformer/posembed_input/pos_embedding'][0,0:self.sequence_length,:])
  
    self.layers[pos_embedding].set_weights(w)



  def call(self, inputs):
 
    inputs = tf.concat([inputs[0],inputs[1]],axis=-1)
    reshape = self.InitialReshape(inputs)
    patches = self.InitialDeconv(reshape)
    projection = self.Projection(patches)
    reshaped_projection = self.Reshape(projection)
    embedding = self.PositionEmbedding(reshaped_projection)

    for i in range(self.transformer_layers):
        
        x1 = self.Norm0[i](embedding)
        attention = self.Attention[i](x1,x1)
        x2 = self.Add0[i]([attention,embedding])
        x3 = self.Norm1[i](x2)
        x3 = self.Dense0[i](x3)
        x3 = self.Dropout0[i](x3)
        x3 = self.Dense1[i](x3)
        x3 = self.Dropout1[i](x3)
        embedding = self.Add1[i]([x3,x2])

    x = self.Flatten(embedding)
    x = self.MapDense(x)
    x = self.MapReshape0(x)
    x = self.MapDeconv(x)
    x = self.MapReshape1(x)

    return x


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

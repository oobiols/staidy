import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn

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

class ResidualBlock(keras.layers.Layer):
    def __init__(self,
                 filters=16,
                 kernel_size=5,
                 **kwargs):

        super(ResidualBlock,self).__init__(**kwargs)
        
        self.Conv1 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=(kernel_size,kernel_size),
                                         strides=(1,1),
                                         activation=tf.nn.leaky_relu,
                                         padding="same")

        self.BN1 = keras.layers.BatchNormalization(axis=-1)

        self.Conv2 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=(kernel_size,kernel_size),
                                         strides=(1,1),
                                         activation=tf.nn.leaky_relu,
                                         padding="same")

        self.BN2 = keras.layers.BatchNormalization(axis=-1)

        self.Conv3 = keras.layers.Conv2D(filters=filters,
                                         kernel_size=(kernel_size,kernel_size),
                                         strides=(1,1),
                                         activation=tf.nn.leaky_relu,
                                         padding="same")

        self.BN3 = keras.layers.BatchNormalization(axis=-1)

        self.Add = keras.layers.Add()

    def call(self,inputs):

       x1 = self.Conv1(inputs)
       x = self.BN1(x1)
       x = self.Conv2(x)
       x = self.BN2(x)
       x = self.Conv3(x)
       x = self.BN3(x)
       x = self.Add([x,x1])

       return x

class NSAttention(NSModelPinn):
  def __init__(self, 
               image_size=[64,256,6],
               filters=[4,16,64],
               factor = 4,
               query_dimension = 100,
               value_dimension = 100,
               **kwargs):

    super(NSAttention, self).__init__(**kwargs)
    self.f = factor
    self.query_dim = query_dimension
    self.value_dim = value_dimension
    self.HR_size = image_size
    self.LR_size = [int(image_size[0]/self.f),int(image_size[1]/self.f)]

    self.res_block = []
    self.pooling = []
    self.invpooling = []
    self.concatenate = []

    self.upsample = keras.layers.UpSampling2D(size=(self.f,self.f),interpolation="bilinear")
    self.downsample = keras.layers.AveragePooling2D(pool_size=(self.f,self.f),strides=(self.f,self.f),padding="same")

    self.conv = keras.layers.Conv2D(filters=16,kernel_size=(4,4),strides=(4,4),padding="same",activation=tf.nn.leaky_relu)
    self.deconv = keras.layers.Conv2DTranspose(filters=16,kernel_size=(4,4),strides=(4,4),padding="same",activation=tf.nn.leaky_relu)

    for i in filters:
        self.res_block.append(ResidualBlock(filters=i,kernel_size=5))
        self.concatenate.append(keras.layers.Concatenate(axis=-1))

    for i in reversed(filters):
        self.res_block.append(ResidualBlock(filters=i,kernel_size=5))

    for i in range(len(filters)-1):
        self.pooling.append(keras.layers.AveragePooling2D(pool_size=(4,8),strides=(4,8),padding="same"))
        self.invpooling.append(keras.layers.UpSampling2D(size=(4,8),interpolation="bilinear"))

    self.extractor = ResidualBlock(filters=4,kernel_size=3) 

    self.query_encoding = keras.layers.Conv2D(filters=query_dimension,
                                                kernel_size=(self.LR_size[0],self.LR_size[1]),
                                                strides=(self.LR_size[0],self.LR_size[1]),
                                                padding="same")

    self.value_encoding = keras.layers.Conv2D(filters=value_dimension,
                                                kernel_size=(4,4), 
                                                strides=(4,4),
                                                padding="same")


    self.attention = keras.layers.Attention()
    self.reshape_query = keras.layers.Reshape((-1,self.query_dim))
    self.reshape_value = keras.layers.Reshape((-1,self.value_dim))
    self.concatenate_coordinates = keras.layers.Concatenate(axis=-1)

  def call(self, inputs):

    low_res_true = inputs[0]
    coordinates = inputs[1]

    x1 = self.upsample(low_res_true)
    x1 = self.res_block[0](x1)
    x1 = self.concatenate_coordinates([x1,coordinates])
    x2 = self.pooling[0](x1)
    x2 = self.res_block[1](x2)
    x3 = self.pooling[1](x2)
    x3 = self.res_block[2](x3)
    y3 = self.invpooling[0](x3)
    y3 = self.res_block[3](y3)
    y3 = self.concatenate[0]([y3,x2])
    y2 = self.invpooling[1](y3)
    y2 = self.res_block[4](y2)
    y2 = self.concatenate[1]([y2,x1])
    high_res_pred = self.res_block[5](y2)


    ##low_res_pred = tf.image.resize(high_res_pred,
    #                                size = self.LR_size,
    #                                method='bilinear',
    #                                preserve_aspect_ratio = True)

    low_res_pred = self.downsample(high_res_pred)
    diff = tf.math.square(low_res_pred-low_res_true)

    query = self.extractor(diff)
    value   = self.extractor(high_res_pred)

    query_enc = self.query_encoding(query)
    value_enc = self.value_encoding(value)


    query_enc = self.reshape_query(query_enc)
    value_enc = self.reshape_value(value_enc)

    attention , scores = self.attention([query_enc,value_enc],return_attention_scores=True)  

    return high_res_pred , low_res_pred


  def compute_data_pde_losses(self, low_res_true, high_res_xz):

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(high_res_xz)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(high_res_xz)

        high_res_pred , low_res_pred = self([low_res_true,high_res_xz])

        u_pred_LR       = low_res_pred[:,:,:,0]
        v_pred_LR       = low_res_pred[:,:,:,1]
        p_pred_LR       = low_res_pred[:,:,:,2]
        nu_pred_LR       = low_res_pred[:,:,:,3]

        u_pred_HR       = high_res_pred[:,:,:,0]
        v_pred_HR       = high_res_pred[:,:,:,1]
        p_pred_HR       = high_res_pred[:,:,:,2]
        nu_pred_HR      = high_res_pred[:,:,:,3]

      # 1st order derivatives
      u_grad   = tape1.gradient(u_pred_HR, high_res_xz)
      v_grad   = tape1.gradient(v_pred_HR, high_res_xz)
      p_grad   = tape1.gradient(p_pred_HR, high_res_xz)
      u_x, u_z = u_grad[:,:,:,0], u_grad[:,:,:,1]
      v_x, v_z = v_grad[:,:,:,0], v_grad[:,:,:,1]
      p_x, p_z = p_grad[:,:,:,0], p_grad[:,:,:,1]
      del tape1

    # 2nd order derivatives
    u_xx = tape2.gradient(u_x, high_res_xz)[:,:,:,0]
    u_zz = tape2.gradient(u_z, high_res_xz)[:,:,:,1]
    v_xx = tape2.gradient(v_x, high_res_xz)[:,:,:,0]
    v_zz = tape2.gradient(v_z, high_res_xz)[:,:,:,1]
    del tape2

    uMse = tf.reduce_mean(tf.square(u_pred_LR - low_res_true[:,:,:,0]))
    vMse = tf.reduce_mean(tf.square(v_pred_LR - low_res_true[:,:,:,1]))
    pMse = tf.reduce_mean(tf.square(p_pred_LR -  low_res_true[:,:,:,2]))
    nuMse = tf.reduce_mean(tf.square(nu_pred_LR - low_res_true[:,:,:,3]))

    # pde error, 0 continuity, 1-2 NS
    pde0    = u_x + v_z
    z = tf.zeros(tf.shape(pde0),dtype=tf.float32)
    pde0Mse    = tf.reduce_mean(tf.square(pde0-z))

    pde1    = u_pred*u_x + v_pred*u_z + p_x - (0.01+ nu_pred)*(1/6000)*(u_xx + u_zz)
    pde1Mse    = tf.reduce_mean(tf.square(pde1-z))

    pde2    = u_pred*v_x + v_pred*v_z + p_z - (0.01 + nu_pred)*(1/6000)*(v_xx + v_zz)
    pde2Mse    = tf.reduce_mean(tf.square(pde2-z))

    return uMse, vMse, pMse, nuMse, pde0Mse, pde1Mse, pde2Mse

  def test_step(self, data):

    inputs = data[0]
 
    high_res_true = inputs[:,:,:,0:4]
    high_res_xz = inputs[:,:,:,4:6]

    low_res_true = tf.image.resize(high_res_true,
                                    size=self.LR_size,
                                    method="bilinear",
                                    preserve_aspect_ratio=True)


    
    high_res_pred , _ = self([low_res_true,high_res_xz])

    u_pred_HR       = high_res_pred[:,:,:,0]
    v_pred_HR       = high_res_pred[:,:,:,1]
    p_pred_HR       = high_res_pred[:,:,:,2]
    nu_pred_HR       = high_res_pred[:,:,:,3]

    uMse = tf.reduce_mean(tf.square(u_pred_HR -   high_res_true[:,:,:,0]))
    vMse = tf.reduce_mean(tf.square(v_pred_HR -   high_res_true[:,:,:,1]))
    pMse = tf.reduce_mean(tf.square(p_pred_HR -   high_res_true[:,:,:,2]))
    nuMse = tf.reduce_mean(tf.square(nu_pred_HR - high_res_true[:,:,:,3]))

      # replica's loss, divided by global batch size
    data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

    loss = data_loss

    #loss += tf.add_n(self.losses)
    # track loss and mae
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    contMse=0.0
    momxMse = 0.0
    momzMse =0.0
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

  def train_step(self, data):

    inputs = data[0]
 
    high_res_true = inputs[:,:,:,0:4]
    high_res_xz = inputs[:,:,:,4:6]

    #low_res_true = tf.image.resize( high_res_true,
    #                                size=self.LR_size,
    #                                method="bilinear",
    #                                preserve_aspect_ratio=True)
    low_res_true = tf.keras.layers.AveragePooling2D(pool_size=(self.f,self.f),strides=(self.f,self.f),padding="same")(high_res_true)

    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(low_res_true, high_res_xz)
      # replica's loss, divided by global batch size
      data_loss  = 0.25*(uMse   + vMse   + pMse + nuMse) 

      loss = data_loss + self.beta[0]*contMse + self.beta[1]*momxMse + self.beta[2]*momzMse
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

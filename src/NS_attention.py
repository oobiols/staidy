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

class ResidualBlockAttentionModule(keras.layers.Layer):
    def __init__(self,
                 f=16,
                 r=4,
                 height=64,
                 width=256,
                 **kwargs):

        super(ResidualBlockAttentionModule,self).__init__(**kwargs)
        
        self.SA = SpatialAttentionModule(f=f,r=r)
        self.elementwise_1 = keras.layers.Multiply()
        self.CA = ChannelAttentionModule(f=f,r=r,height=height,width=width)
        self.elementwise_2 = keras.layers.Multiply()
        self.res = keras.layers.Add()

    def call(self,inputs):
        
       spatial = self.SA(inputs)
       spatial_attention = self.elementwise_1([spatial,inputs])
       channel = self.CA(spatial_attention)
       channel_attention = self.elementwise_2([channel,spatial_attention])
       x = self.res([inputs,channel_attention])
      
       return x , spatial

class SpatialAttentionModule(keras.layers.Layer):
    def __init__(self,
                 f=16 ,
                 r=4 ,
                 **kwargs):

        super(SpatialAttentionModule,self).__init__(**kwargs)
        
        c = int(f/r)
        self.reduction = keras.layers.Conv2D(filters=c,kernel_size=(1,1),strides=(1,1),padding="same",activation=tf.nn.leaky_relu)
        self.dilated = keras.layers.Conv2D(filters=c,kernel_size=(5,5),strides=(1,1),padding="same",activation=tf.nn.leaky_relu)
        self.spatial = keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=(1,1),padding="same",activation=tf.nn.leaky_relu)
        

    def call(self,inputs):

       x = self.reduction(inputs)
       x = self.dilated(x)
       x = self.spatial(x)
      
       return x

class ChannelAttentionModule(keras.layers.Layer):
    def __init__(self,
                 f=16,
                 r = 4 ,
                 height = 16,
                 width  = 64, 
                 **kwargs):

        super(ChannelAttentionModule,self).__init__(**kwargs)
        
        c = int(f/r)
        self.pooling = keras.layers.AveragePooling2D(pool_size=(height,width),strides=(height,width))
        self.mlp1 = keras.layers.Dense(c,activation=tf.nn.leaky_relu)
        self.mlp2 = keras.layers.Dense(f,activation=tf.nn.leaky_relu)
         
    def call(self,inputs):

       x = self.pooling(inputs)
       x = self.mlp1(x)
       x = self.mlp2(x)
      
       return x

class NSAttention(NSModelPinn):
  def __init__(self, 
               image_size=[64,256,6],
               filters=[4,16,64],
               factor = 4,
               strides= 2,
               kernel_size = 5,
               num_attention = 1,
               attention = True,
               **kwargs):

    super(NSAttention, self).__init__(**kwargs)
    self.f = factor
    self.HR_size = image_size
    self.LR_size = [int(image_size[0]/self.f),int(image_size[1]/self.f)]
    self.num_attention = num_attention
    self.attention = attention
    self.feature_extractor = []
    self.reconstruction = []
    self.RBAM = []

    self.k = kernel_size
    self.s = strides

    self.upsample = keras.layers.UpSampling2D(size=(self.f,self.f),interpolation="bilinear")
    self.concatenate_coordinates = keras.layers.Concatenate(axis=-1)

  
    for i in filters:
        self.feature_extractor.append(keras.layers.Conv2D(filters=i,
                                             kernel_size=(self.k,self.k),
                                             strides=(self.s,self.s),
						padding="same",
                                             activation=tf.nn.leaky_relu))
    
    for i in reversed(filters):
        self.reconstruction.append(keras.layers.Conv2DTranspose(filters=i,           
                                             kernel_size=(self.k,self.k),
                                             strides=(self.s,self.s),
					     padding="same",
                                             activation=tf.nn.leaky_relu))
 
    fil = filters[-1]  

    for i in range(self.num_attention):
         self.RBAM.append(ResidualBlockAttentionModule( f=fil,
                                                        r=4,
                                                        height=self.HR_size[0],
                                                        width=self.HR_size[1]))


  def call(self, inputs):

    low_res_true = inputs[0]
    coordinates = inputs[1]/500

    up = self.upsample(low_res_true)
    x1 = self.concatenate_coordinates([up,coordinates])
    for layer in self.feature_extractor:
        x1 = layer(x1)
    
    x2 = x1
    if self.attention == True:
     for layer in self.RBAM:
        x2 , spatial_attended_map = layer(x2)

    x3 = x2
    for layer in self.reconstruction:
        x3 = layer(x3)


    high_res_pred = x3
    low_res_pred = tf.image.resize(high_res_pred,
                                   size = self.LR_size,
                                   method='bilinear',
                                   preserve_aspect_ratio = False)

    return coordinates , low_res_pred, spatial_attended_map


  def compute_data_pde_losses(self, high_res_true, high_res_xz,labels):

    low_res_true = tf.image.resize( high_res_true,
                                    size=self.LR_size,
                                    method="bilinear",
                                    preserve_aspect_ratio=True)

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

    uMse = tf.reduce_mean(tf.square(u_pred_LR - low_res_true[:,:,:,0])) \
           + tf.reduce_mean(tf.square(u_pred_HR[:,59:64,:]-high_res_true[:,59:64,:,0]))\
           + tf.reduce_mean(tf.square(u_pred_HR[:,:,0]-high_res_true[:,:,0,0]\
           + tf.reduce_mean(tf.square(u_pred_HR[:,:,-1]-high_res_true[:,:,-1,0]\

    vMse = tf.reduce_mean(tf.square(v_pred_LR - low_res_true[:,:,:,1])) \
           + tf.reduce_mean(tf.square(v_pred_HR[:,59:64,:]-high_res_true[:,59:64,:,1]))\
           + tf.reduce_mean(tf.square(v_pred_HR[:,:,0]-high_res_true[:,:,0,1]\
           + tf.reduce_mean(tf.square(v_pred_HR[:,:,-1]-high_res_true[:,:,-1,1]\

    pMse = tf.reduce_mean(tf.square(p_pred_LR - low_res_true[:,:,:,2])) \
           + tf.reduce_mean(tf.square(p_pred_HR[:,59:64,:]-high_res_true[:,59:64,:,2]))\
           + tf.reduce_mean(tf.square(p_pred_HR[:,:,0]-high_res_true[:,:,0,2]\
           + tf.reduce_mean(tf.square(p_pred_HR[:,:,-1]-high_res_true[:,:,-1,2]\

    nuMse = tf.reduce_mean(tf.square(nu_pred_LR - low_res_true[:,:,:,3])) \
           + tf.reduce_mean(tf.square(nu_pred_HR[:,59:64,:]-high_res_true[:,59:64,:,3]))\
           + tf.reduce_mean(tf.square(nu_pred_HR[:,:,0]-high_res_true[:,:,0,3]\
           + tf.reduce_mean(tf.square(nu_pred_HR[:,:,-1]-high_res_true[:,:,-1,3]\


    # pde error, 0 continuity, 1-2 NS
    pde0    = u_x + v_z
    z = tf.zeros(tf.shape(pde0),dtype=tf.float32)
    pde0Mse    = tf.reduce_mean(tf.square(pde0-z))

    pde1    = u_pred_HR*u_x + v_pred_HR*u_z + p_x - (0.01+ nu_pred_HR)*(1/(6000*500))*(u_xx + u_zz)
    pde1Mse    = tf.reduce_mean(tf.square(pde1-z))
    #pde1Mse = 0 


   # pde2    = u_pred_HR*v_x + v_pred_HR*v_z + p_z - (0.01 + nu_pred_HR)*(1/6000)*(v_xx + v_zz)
   # pde2Mse    = tf.reduce_mean(tf.square(pde2-z))
    pde2Mse = 0 

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
    data_loss  = 0.5*(uMse   + vMse )#  + pMse + nuMse) 

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
    labels = data[1]
 
    high_res_true = inputs[:,:,:,0:4]
    high_res_xz = inputs[:,:,:,4:6] 
    

    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(high_res_true, high_res_xz,labels)
      # replica's loss, divided by global batch size
      data_loss  = 0.5*(uMse   + vMse)# + pMse + nuMse) 

      
      beta_cont = int(data_loss.numpy()/contMse.numpy())
      #beta_momx = int(data_loss.numpy()/momxMse.numpy())
      beta_momx = int(data_loss/momxMse.numpy())

      loss = data_loss + self.beta[0]*beta_cont*contMse + self.beta[1]*beta_momx*momxMse + self.beta[2]*momzMse

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

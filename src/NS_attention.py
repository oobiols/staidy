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
                                         kernel_size=(1,1),
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
                                         kernel_size=(1,1),
                                         strides=(1,1),
                                         activation=tf.nn.leaky_relu,
                                         padding="same")

        self.BN3 = keras.layers.BatchNormalization(axis=-1)

        self.Add = keras.layers.Add()

    def call(self,inputs):

       x1 = self.Conv1(inputs)
       #x = self.BN1(x1)dd
       x = self.Conv2(x1)
       #x = self.BN2(x)
       x = self.Conv3(x)
       #x = self.BN3(x)
       x = self.Add([x,x1])

       return x

class Mlp(keras.layers.Layer):
    def __init__(self,
                 hidden_units = [10,10],
                 dropout_rate = 0,
                 **kwargs):

        super(Mlp,self).__init__(**kwargs)
        
        self._layers = []

        for i in hidden_units:    
            self._layers.append( keras.layers.Dense(i,activation=tf.nn.leaky_relu))
            self._layers.append( keras.layers.Dropout(dropout_rate))

    def call(self,inputs):
        x = inputs
        for layer in self._layers:
            x  = layer(x)
        return x


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self,
                 num_heads = 2,
                 proj_dimension = 32,
                 **kwargs):

        super(MultiHeadAttention,self).__init__(**kwargs)
        
        self.transformer_units=[proj_dimension*2,proj_dimension]

        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha         = keras.layers.MultiHeadAttention(num_heads=num_heads, 
                                                    key_dim=proj_dimension, 
                                                    dropout=0.1) 
        self.add_1         = keras.layers.Add()
        self.layernorm_2   = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp           = Mlp(hidden_units=self.transformer_units,dropout_rate=0)
        self.add_2         = keras.layers.Add()


    def call(self,inputs):

       x = self.layernorm_1(inputs)
       attention_output, attention_scores= self.mha(x,x,return_attention_scores=True)
       x2 = self.add_1([attention_output,inputs])
       x3 = self.layernorm_2(x2)
       x3 = self.mlp(x2)
       x  = self.add_2([x3,x2])

       return x , attention_scores


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

class NSSelfAttention(NSModelPinn):
  def __init__(self, 
               image_size = [32,128,6],
               patch_size = [4,16],
               filters=[16,32],
               kernel_size = 5,
               num_attention = 1,
               num_heads=2,
               proj_dimension=32,
               **kwargs):

    super(NSSelfAttention, self).__init__(**kwargs)

    rows_image    = image_size[0]
    columns_image = image_size[1]


    rows_patch    = patch_size[0]
    columns_patch = patch_size[1]

    self._channels = image_size[2]
    self._pixels_patch = rows_patch*columns_patch
    self._n_patch_y = rows_image//rows_patch
    self._n_patch_x = columns_image//columns_patch


    self.k = kernel_size
    self._res_blocks = []
    self._self_attention = []
    self._n_res_blocks = len(filters)
    self._num_heads = num_heads
    self._proj_dimension = proj_dimension
    self._rows_patch = rows_patch
    self._columns_patch = columns_patch

    for i in filters:
        self._res_blocks.append(ResidualBlock(i,kernel_size=self.k))

    
    self.patches = keras.layers.Conv2D(filters=proj_dimension,
                                                kernel_size = patch_size,
                                                strides = patch_size,
                                                activation = tf.nn.leaky_relu)


    self.reshape = keras.layers.Reshape((-1,proj_dimension))
    for _ in range(num_attention):
        self._self_attention.append(MultiHeadAttention(num_heads=self._num_heads,proj_dimension=self._proj_dimension))

   
    self.reshape_1 = keras.layers.Reshape((self._n_patch_y*self._n_patch_x,self._pixels_patch,self._channels))
    self.conv = keras.layers.Conv2D
  def call(self, inputs):

    features = tf.concat([inputs[0],inputs[1]],axis=-1)
    for layer in self._res_blocks:
     features = layer(features)


    enc_patches = self.patches(features)
    enc_patches = self.reshape(enc_patches)
    

    for layer in self._self_attention:
     enc_patches , scores = layer(enc_patches)

    r_scores = tf.reduce_sum(scores,axis=1,keepdims=False)
    r_scores = tf.reduce_sum(r_scores,axis=1,keepdims=False)
    r_scores = tf.divide(tf.subtract(r_scores,tf.reduce_min(r_scores)),tf.subtract(tf.reduce_max(r_scores),tf.reduce_min(r_scores)))
    bin_of_patch = tf.histogram_fixed_width_bins(r_scores, value_range=[0.0,1.0], nbins=3)[0]
    hi = tf.less(bin_of_patch,1)
    hi = tf.expand_dims(hi,axis=1)
    zeroidx = tf.where(hi)[:,0]
    patches = tf.gather(r_scores,zeroidx,axis=1)


   

    
    return enc_patches


  def compute_data_pde_losses(self, low_res_true, low_res_xz,labels):


    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(low_res_xz)
      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(low_res_xz)

        enc_patches = self([low_res_true,low_res_xz])


    uMse = tf.reduce_mean(enc_patches)
    vMse = 0
    pMse = 0
    nuMse = 0
    pde0Mse = 0 
    pde1Mse = 0
    pde2Mse = 0
    return uMse, vMse, pMse, nuMse, pde0Mse, pde1Mse, pde2Mse


  def test_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:4]
    low_res_xz = inputs[:,:,:,4:6]


    #scores = self([low_res_true,low_res_xz])


#    uMse = tf.reduce_mean(tf.square(u_pred_HR -   high_res_true[:,:,:,0]))
#    vMse = tf.reduce_mean(tf.square(v_pred_HR -   high_res_true[:,:,:,1]))
#    pMse = tf.reduce_mean(tf.square(p_pred_HR -   high_res_true[:,:,:,2]))
#    nuMse = tf.reduce_mean(tf.square(nu_pred_HR - high_res_true[:,:,:,3]))

      # replica's loss, divided by global batch size
    uMse = 0
    vMse = 0 
    pMse = 0
    nuMse = 0
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
 
    low_res_true = inputs[:,:,:,0:4]
    low_res_xz = inputs[:,:,:,4:6] 
    

    with tf.GradientTape(persistent=True) as tape0:
      # compute the data loss for u, v, p and pde losses for
      # continuity (0) and NS (1-2)
      uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse = \
        self.compute_data_pde_losses(low_res_true, low_res_xz,labels)
      # replica's loss, divided by global batch size
      data_loss  = 0.5*(uMse   + vMse)# + pMse + nuMse) 

      
      #beta_cont = int(data_loss.numpy()/contMse.numpy())
      #beta_momx = int(data_loss/momxMse.numpy())
      beta_momx = 0
      beta_cont = 0

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

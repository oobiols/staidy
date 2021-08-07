import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn
from tensorflow.python.ops import math_ops

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


class ScorerNetwork(keras.layers.Layer):
    def __init__(self,
                  scorer_filters=[8,16,32],
                  kernel_size = 3,
                  strides = 1,
                  patch_size = [8,32],
                  **kwargs):

        super(ScorerNetwork,self).__init__(**kwargs)

        self._filters= scorer_filters
        self._k = kernel_size
        self._s = strides
        self._patch_size = patch_size
        self._conv = []

        for f in self._filters:

         self._conv.append(keras.layers.Conv2D(filters=f,
                                                kernel_size = self._k,
                                                strides = self._s,
                                                activation= tf.nn.leaky_relu,
                                                padding="same") )


        self._spatial_attention = keras.layers.Conv2D(filters=1,
                                                kernel_size =self._k,
                                                strides = self._s,
                                                activation=tf.nn.leaky_relu,
                                                padding="same")

        self._pool = keras.layers.MaxPooling2D(pool_size=self._patch_size, 
                                               strides=self._patch_size, 
                                               padding="same")

    
        self._flatten = keras.layers.Reshape((-1,1))
        self._softmax = keras.layers.Softmax(axis=1)

    def call(self,inputs):

     x = inputs
     for layer in self._conv:
      x = layer(x)

     spatial_attention = self._spatial_attention(x)
     score_matrix = self._pool(spatial_attention)
     flattened_scores = self._flatten(score_matrix)
     scores = self._softmax(flattened_scores)
     scores = self._MinMaxScaler(scores)

     return spatial_attention , scores 

    def _MinMaxScaler(self,x):
 
     max = tf.reduce_max(x)
     min = tf.reduce_min(x)
 
     x = tf.divide(x-min,max-min)
 
     return x


class RankingModule():
    def __init__(self,
                    nbins,
                    batch_size=1,
                    n_patches = 16,
                    self_attention=False,
                    **kwargs):

        super(RankingModule,self).__init__(**kwargs)

        self._self_attention = self_attention
        self._n_bins = nbins
        self._batch_size = batch_size
        self._n_patches = n_patches

    def __call__(self,inputs):
     
        scores = inputs

        if (self._self_attention):
         scores = self.reduce_scores(scores)

        scores = tf.reshape(scores,shape=(self._batch_size*self._n_patches))

        bin_per_patch = self.find_bins(scores)
        indices = []
        for i in range(1,self._n_bins+1):

         idx  = self.get_patches_bin(bin_per_patch,i)
         indices.append(idx)

        return indices

    def get_patches_bin(self,bin_per_patch,bin_number):
        
        bin = tf.equal(bin_per_patch,bin_number)
        i = tf.where(bin)

        return i

    def find_bins(self,scores):

     bins = np.linspace(0,1.01,self._n_bins+1).tolist()
     bin_per_patch = math_ops._bucketize(scores, boundaries=bins)

     return bin_per_patch

    def reduce_scores(self,scores):

      # finding per patch score (finding from more to least important patches)
      scores = tf.reduce_sum(scores,axis=1,keepdims=False)
      scores = tf.reduce_sum(scores,axis=1,keepdims=False)

      # min max scaling between 0 and 1
      scores = tf.divide(tf.subtract(scores,tf.reduce_min(scores)),tf.subtract(tf.reduce_max(scores),tf.reduce_min(scores)))
      return scores

class NSAmrScorer(NSModelPinn):
  def __init__(self, 
               image_size = [32,128,6],
               patch_size = [4,16],
               scorer_filters=[3,16,32],
               scorer_kernel_size = 5,
               batch_size=1,
               nbins = 4,
               **kwargs):

    super(NSAmrScorer, self).__init__(**kwargs)

    self._batch_size = batch_size
    self._rows_image    = image_size[0]
    self._columns_image = image_size[1]


    self._rows_patch = patch_size[0]
    self._columns_patch = patch_size[1]

    self._channels_in = image_size[2]
    self._channels_coord = 2
    self._channels_out = self._channels_in - self._channels_coord
    self._pixels_patch = self._rows_patch*self._columns_patch
    self._n_patch_y = self._rows_image//self._rows_patch
    self._n_patch_x = self._columns_image//self._columns_patch
    self._n_patches = self._n_patch_y * self._n_patch_x
    self._n_bins = nbins

    self._coord_upsampling = []
    self._decoder = []
    self._output_deconv = []

    self._scorer = ScorerNetwork(scorer_filters= scorer_filters,
                                kernel_size= scorer_kernel_size,
                                patch_size= patch_size)


    self._ranker = RankingModule(self._n_bins,
                                 self._batch_size,
                                 self._n_patches,
                                 self_attention=False)


    for filter in reversed(scorer_filters):
        self._decoder.append(keras.layers.Conv2DTranspose(filters=filter,
                                                                kernel_size=5,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))


    for _ in range(self._n_bins-1):

     self._output_deconv.append(keras.layers.Conv2DTranspose(filters=1,
                                                                kernel_size=5,
                                                                strides=2,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))
       
    level=1
    for _ in range(self._n_bins):
     self._coord_upsampling.append(keras.layers.UpSampling2D(size=level,interpolation="bilinear"))
     level=2*level

  def _ranking(self,scores):

        scores = tf.reshape(scores,shape=(self._batch_size*self._n_patches,1))
        bins = np.linspace(0,1.01,self._n_bins+1).tolist()
        bin_per_patch = math_ops._bucketize(scores, boundaries=bins)

        indices = []

        for i in range(1,self._n_bins+1):

         bin = tf.equal(bin_per_patch,i)
         idx = tf.where(bin[:,0])
         indices.append(idx)

        return indices




  def call(self, inputs):

    features = inputs[0]
    coordinates = inputs[1]
   
    features , scores = self._scorer(features) #scores shape [BS,NP]

    I     = self._ranking(scores) #indices shape [BS*NP]

    patches     = self.from_image_to_patch_sequence(features) #patches shape [BS*NP,h,w,C]
    coordinates = self.from_image_to_patch_sequence(coordinates) #patches shape [BS*NP,h,w,C]

    P = []
    XZ = []
    for i , idx in enumerate(I):
     p = tf.squeeze(tf.gather(patches,idx,axis=0),axis=1)
     for j in range(i):
      p = self._output_deconv[j](p)

     xz = tf.squeeze(tf.gather(coordinates,idx,axis=0),axis=1)
     xz = self._coord_upsampling[i](xz)
     p = tf.concat([p,xz],axis=-1)
     for layer in self._decoder:
      p = layer(p)

     P.append(p)
     XZ.append(xz)

    return P, I, XZ 

  def compute_data_loss(self,true_patches, predicted_patches,indices):

    uMse=0.0
    vMse=0.0
    pMse=0.0
    cont = 0.0 
    lr_predicted_patches = predicted_patches[0]

    for i, idx in enumerate(indices):

        true_p = tf.gather(true_patches,idx,axis=0)
        true_p = tf.reshape(true_p,
                            shape=(true_p.shape[0],
                                    lr_predicted_patches.shape[1],
                                    lr_predicted_patches.shape[2],
                                    lr_predicted_patches.shape[3]))
        pred_p = predicted_patches[i]
        pred_p = tf.image.resize(pred_p,
                                 size =(self._rows_patch,self._columns_patch),
                                 method='bicubic')
        
        if true_p.shape[0] > 0: 

            cont += 1
            uMse += tf.reduce_mean(tf.square(true_p[:,:,:,0] - pred_p[:,:,:,0]))
            vMse += tf.reduce_mean(tf.square(true_p[:,:,:,1] - pred_p[:,:,:,1]))
            pMse += tf.reduce_mean(tf.square(true_p[:,:,:,2] - pred_p[:,:,:,2]))


    return (1/cont)*uMse, (1/cont)*vMse, (1/cont)*pMse

  def compute_pde_loss(self,u_grad,v_grad):

    contMse = 0
    cont = 0.0
    for i , _ in enumerate(u_grad):
     grad = u_grad[i]
     ux = grad[:,:,:,0]
     grad = v_grad[i]
     vz = grad[:,:,:,1]
     if ux.shape[0]>0:
      contMse += tf.reduce_mean(tf.square(ux+vz))
      cont = cont+1

    return (1/cont)*contMse

  def from_image_to_patch_sequence(self, x):

    channels = x.shape[-1]
  
    x = tf.image.extract_patches( images = x, 
                                            sizes = [1,self._rows_patch,self._columns_patch,1],
                                            strides = [1,self._rows_patch,self._columns_patch,1],
                                            rates=[1,1,1,1],
                                            padding="VALID")

    x = tf.reshape(x,
                   shape=(self._batch_size*self._n_patches,
                            self._rows_patch,
                            self._columns_patch,
                            channels))

    return x               

  def compute_loss(self, low_res_true, low_res_xz):

    
    true_patches = self.from_image_to_patch_sequence(low_res_true)
    XZ = low_res_xz

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(XZ)

        predicted_patches, indices, XZ = self([low_res_true,XZ])
 
        u_pred_patches = []
        v_pred_patches = []
        p_pred_patches = []
        for p in predicted_patches:
            u_pred_patches.append(p[:,:,:,0])
            v_pred_patches.append(p[:,:,:,1])
            p_pred_patches.append(p[:,:,:,2])

    u_grad = tape1.gradient(u_pred_patches,XZ)
    v_grad = tape1.gradient(v_pred_patches,XZ)


    uMse, vMse, pMse = self.compute_data_loss(true_patches, predicted_patches,indices)

    contMse = self.compute_pde_loss(u_grad,v_grad)

    return uMse, vMse, pMse, contMse


  def test_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:3]
    low_res_xz = inputs[:,:,:,3:5]

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, contMse = self.compute_loss(low_res_true, low_res_xz)

      data_loss  = (1/3)*(uMse   + vMse + pMse)# + pMse + nuMse) 

      cont_loss = contMse
      
      beta_cont = int(data_loss/contMse)
      loss = data_loss + self.beta[0]*beta_cont*cont_loss

    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(contMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat

  def train_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:3]
    low_res_xz = inputs[:,:,:,3:5] 

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, contMse = self.compute_loss(low_res_true, low_res_xz)

      data_loss  = (1/3)*(uMse   + vMse + pMse)# + pMse + nuMse) 

      cont_loss = contMse
      
      beta_cont = int(data_loss/contMse)
      loss = data_loss + self.beta[0]*beta_cont*cont_loss

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
    self.trainMetrics['cont_loss'].update_state(cont_loss)
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat

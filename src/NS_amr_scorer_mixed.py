import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn
from tensorflow.python.ops import math_ops

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

class UpSampling2DBilinear(keras.layers.Layer):
    def __init__(self,
		 size=(32,128),
                 **kwargs):

        super(UpSampling2DBilinear,self).__init__(**kwargs)

        self._size = size

    def call(self,inputs):

         x = inputs
         x = tf.image.resize(x, size=self._size,method='bilinear')

         return x

class UpSampling2DBicubic(keras.layers.Layer):
    def __init__(self,
		 size=(32,128),
                 **kwargs):

        super(UpSampling2DBicubic,self).__init__(**kwargs)

        self._size = size

    def call(self,inputs):

         x = inputs
         x = tf.image.resize(x, size=self._size,method='bicubic')

         return x



class NSAmrScorer(NSModelPinn):
  def __init__(self, 
               image_size = [32,128,6],
               patch_size = [4,16],
               scorer_filters=[3,16,32],
               filters = [3,16,32,128],
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
    self._channels_out = 4
    self._pixels_patch = self._rows_patch*self._columns_patch
    self._n_patch_y = self._rows_image//self._rows_patch
    self._n_patch_x = self._columns_image//self._columns_patch
    self._n_patches = self._n_patch_y * self._n_patch_x
    self._n_bins = nbins
    self._upsampling = []
    self._coord_upsampling = []
    self._decoder = []
    self._encoder = []
    self._enc_dec = []

    self._proj_dim = self._rows_patch*self._columns_patch*4
    self._scorer = ScorerNetwork(scorer_filters= scorer_filters,
                                kernel_size= scorer_kernel_size,
                                patch_size= patch_size)

    self._filters =  filters
    self._filters[0] = self._channels_out

    for filter in self._filters:
        self._encoder.append(keras.layers.Conv2D(filters=filter,
                                                                kernel_size=3,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))
    rev_filters = self._filters[::-1]

    for i in range(len(rev_filters)):

       if i == (len(rev_filters)-1):
        self._decoder.append(keras.layers.Conv2DTranspose(filters=filter,
                                                                kernel_size=3,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same",
                                                                dtype = 'float32'))
       else:

        self._decoder.append(keras.layers.Conv2DTranspose(filters=filter,
                                                                kernel_size=3,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))
       
    level=1
    for _ in range(self._n_bins):
     r = self._rows_patch * level
     c = self._columns_patch * level
     self._upsampling.append(UpSampling2DBicubic(size=(r,c)))
     self._coord_upsampling.append(UpSampling2DBicubic(size=(r,c)))
     level=2*level

    self._bins = np.linspace(0.0,1.01,self._n_bins+1).tolist()

  def _ranking(self,scores):

        scores = tf.reshape(scores,shape=(self._batch_size*self._n_patches,1))
        
        bin_per_patch = math_ops._bucketize(tf.cast(scores,dtype='float32'), boundaries=self._bins)
        indices = []

        for i in range(1,self._n_bins+1):

         bin = tf.equal(bin_per_patch,i)
         idx = tf.where(bin[:,0])
         indices.append(idx)

        return indices

  def call(self, inputs):

    flowvar = inputs[0]
    coordinates = inputs[1]

    features , scores = self._scorer(flowvar)

    I     = self._ranking(scores) #indices shape [BS*NP]

    features = tf.concat([flowvar,features],axis=-1)

    features     = self.from_image_to_patch_sequence(features) #patches shape [BS*NP,h,w,C]
    coordinates = self.from_image_to_patch_sequence(coordinates) #patches shape [BS*NP,h,w,C]

    u = []
    v = []
    pressure = []
    nu = []
    XZ = []
    for i , idx in enumerate(I):

     p = tf.squeeze(tf.gather(features,idx,axis=0),axis=1)
     p = self._upsampling[i](p)

     xz = tf.squeeze(tf.gather(coordinates,idx,axis=0),axis=1)
     xz = self._coord_upsampling[i](xz)

     p = tf.concat([p,xz],axis=-1)

     for layer in self._encoder:
      p = layer(p)

     for layer in self._decoder:
      p = layer(p)
     
     u.append(p[:,:,:,0])
     v.append(p[:,:,:,1])
     pressure.append(p[:,:,:,2])
     nu.append(p[:,:,:,3])
     XZ.append(xz)

    return u, v, pressure, nu, I, XZ 

  def train_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:4]
    low_res_xz = inputs[:,:,:,4:6] 
    Re_nulaminar = inputs[:,:,:,6:] 

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, nuMse, cont_loss, momx_loss, momy_loss = self.compute_loss(low_res_true, low_res_xz, Re_nulaminar)

      data_loss  = (1/4)*(nuMse + uMse   + vMse + pMse)

      beta_cont = float(data_loss/cont_loss)
      beta_momx = float(data_loss/momx_loss)
      beta_momy = float(data_loss/momy_loss)

      loss = data_loss + self.beta[0]*(beta_cont*cont_loss + beta_momx * momx_loss + beta_momy * momy_loss)
      scaled_loss = self.optimizer.get_scaled_loss(loss)

    lossGrad = tape0.gradient(scaled_loss, self.trainable_variables)
    lossGrad = self.optimizer.get_unscaled_gradients(lossGrad)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['data_loss'].update_state(data_loss)
    self.trainMetrics['cont_loss'].update_state(cont_loss)
    self.trainMetrics['momx_loss'].update_state(momx_loss)
    self.trainMetrics['momy_loss'].update_state(momy_loss)
    self.trainMetrics['nuMse'].update_state(nuMse)
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat

  def compute_loss(self, low_res_true, low_res_xz,Re_nulaminar):
    
    Re_nulaminar = self.from_image_to_patch_sequence(Re_nulaminar)
    XZ = low_res_xz

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(XZ)

      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(XZ)

        u_p,v_p,p_p,nu_p, indices, XZ = self([low_res_true,XZ])

      u_grad = tape1.gradient(u_p,XZ)
      v_grad = tape1.gradient(v_p,XZ)
      p_grad = tape1.gradient(p_p,XZ)
      nu_grad = tape1.gradient(nu_p,XZ)

      del tape1

    u_grad_2 = tape2.gradient(u_grad,XZ)
    v_grad_2 = tape2.gradient(v_grad,XZ)

    del tape2

    contMse, momxMse, momzMse = self.compute_pde_loss(u_p,v_p,p_p,nu_p,u_grad,v_grad,p_grad,nu_grad,u_grad_2,v_grad_2,Re_nulaminar,indices)
    uMse, vMse, pMse , nuMse= self.compute_data_loss(low_res_true, indices, u_p,v_p,p_p,nu_p)

    return uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse

  def compute_data_loss(self,low_res_true, indices, u_p,v_p,p_p,nu_p):

    uMse = 0.0
    vMse = 0.0
    pMse = 0.0
    nuMse = 0.0
    cont = 0.0

    level=1
    low_res_true = self.from_image_to_patch_sequence(low_res_true) #patches shape [BS*NP,h,w,C]

    for i in range(self._n_bins):
     
     true_p = tf.squeeze(tf.gather(low_res_true,indices[i],axis=0),axis=1)
     u_p[i] = tf.keras.layers.AveragePooling2D(pool_size=(level, level), strides=(level,level), padding="same",dtype='float32')(tf.expand_dims(u_p[i],axis=-1))
     v_p[i] = tf.keras.layers.AveragePooling2D(pool_size=(level, level), strides=(level,level), padding="same",dtype='float32')(tf.expand_dims(v_p[i],axis=-1))
     p_p[i] = tf.keras.layers.AveragePooling2D(pool_size=(level, level), strides=(level,level), padding="same",dtype='float32')(tf.expand_dims(p_p[i],axis=-1))
     nu_p[i] = tf.keras.layers.AveragePooling2D(pool_size=(level, level), strides=(level,level), padding="same",dtype='float32')(tf.expand_dims(nu_p[i],axis=-1))
     level = level*2
  
     if true_p.shape[0] > 0: 

      uMse += tf.reduce_mean(tf.square(true_p[:,:,:,0] - u_p[i][:,:,:,0]))
      vMse += tf.reduce_mean(tf.square(true_p[:,:,:,1] - v_p[i][:,:,:,0]))
      pMse += tf.reduce_mean(tf.square(true_p[:,:,:,2] - p_p[i][:,:,:,0]))
      nuMse += tf.reduce_mean(tf.square(true_p[:,:,:,3] - nu_p[i][:,:,:,0]))

     cont += 1
    return (1/cont)*uMse , (1/cont)*vMse , (1/cont)*pMse , (1/cont)*nuMse

  def compute_pde_loss(self,u_p,v_p,p_p,nu_p,u_grad,v_grad,p_grad,nu_grad,u_grad_2,v_grad_2,Re_nulaminar,indices):

    contMse = 0.0
    momxMse = 0.0
    momyMse = 0.0
    cont = 0.0

    level = 1
    for i , _ in enumerate(u_grad):

     if u_grad[i][:,:,:,0].shape[0]>0:

      re_nulaminar = tf.squeeze(tf.gather(Re_nulaminar,indices[i],axis=0),axis=1)
      re_nulaminar = tf.keras.layers.UpSampling2D(size=level,interpolation='nearest',dtype='float32')(re_nulaminar)

      contMse += tf.reduce_mean(tf.square(u_grad[i][:,:,:,0]+v_grad[i][:,:,:,1]))

      momx = u_p[i] * u_grad[i][:,:,:,0] + v_p[i]*u_grad[i][:,:,:,1] + p_grad[i][:,:,:,0] - re_nulaminar[:,:,:,0]*(2*nu_grad[i][:,:,:,0]*u_grad[i][:,:,:,0] + nu_grad[i][:,:,:,1]*(u_grad[i][:,:,:,1] + v_grad[i][:,:,:,0]) + (re_nulaminar[:,:,:,1] + nu_p[i])*(u_grad_2[i][:,:,:,0] + u_grad_2[i][:,:,:,1]))

      momxMse += tf.reduce_mean(tf.square(momx))

      momy = u_p[i] * v_grad[i][:,:,:,0] + v_p[i]*v_grad[i][:,:,:,1] + p_grad[i][:,:,:,1] - re_nulaminar[:,:,:,0]*(2*nu_grad[i][:,:,:,1]*v_grad[i][:,:,:,1] + nu_grad[i][:,:,:,0]*(u_grad[i][:,:,:,1]+ v_grad[i][:,:,:,0]) + (re_nulaminar[:,:,:,1]+nu_p[i])*(v_grad_2[i][:,:,:,0]+v_grad_2[i][:,:,:,1]))

      momyMse += tf.reduce_mean(tf.square(momy))


     level=level*2
     cont = cont+1

    return (1/cont)*contMse, (1/cont)*momxMse, (1/cont)*momyMse

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

  def test_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:4]
    low_res_xz = inputs[:,:,:,4:6] 
    Re_nulaminar = inputs[:,:,:,6:] 

    uMse, vMse, pMse, nuMse, contMse, momxMse, momyMse = self.compute_loss(low_res_true, low_res_xz, Re_nulaminar)

    data_loss  = (1/4)*(nuMse + uMse   + vMse + pMse)

    beta_momx = float(data_loss/momxMse)
    beta_cont = float(data_loss/contMse)
    beta_momy = float(data_loss/momyMse)

    loss = data_loss + self.beta[0]*(beta_cont*contMse + beta_momx*momxMse + beta_momy*momyMse)
    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(contMse)
    self.validMetrics['momx_loss'].update_state(momxMse)
    self.validMetrics['momy_loss'].update_state(momyMse)
    self.validMetrics['nuMse'].update_state(nuMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


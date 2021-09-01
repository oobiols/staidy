import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn
from tensorflow.python.ops import math_ops

class EncoderDecoder(keras.layers.Layer):
    def __init__(self,
		 filters=[3,16,32],
		 strides=1,
                  **kwargs):

        super(EncoderDecoder,self).__init__(**kwargs)

        self._enc = []
        self._dec = []
        self._s = strides

        for filter in filters:
         self._enc.append(keras.layers.Conv2D(filters=filter,kernel_size=5,strides=self._s,padding="same", activation=tf.nn.leaky_relu))
        for filter in reversed(filters):
         self._dec.append(keras.layers.Conv2DTranspose(filters=filter,kernel_size=5,strides=self._s,padding="same", activation=tf.nn.leaky_relu))

    def call(self,inputs):
        x = inputs
        for layer in self._enc:
         x = layer(x)
        for layer in self._dec:
         x = layer(x)
        return x



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

    self.trainMetrics = {}
    self.validMetrics = {}
    names = ['loss','lr_data_loss','lr_pde_loss','hr_data_loss','hr_pde_loss']
    for key in names:
      self.trainMetrics[key] = keras.metrics.Mean(name='train_'+key)
      self.validMetrics[key] = keras.metrics.Mean(name='valid_'+key)


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
    self._upsampling = []
    self._coord_upsampling = []
    self._decoder = []
    self._encoder = []
    self._enc_dec = []
    self._scorer = ScorerNetwork(scorer_filters= scorer_filters,
                                kernel_size= scorer_kernel_size,
                                patch_size= patch_size)

    self._filters =  filters
    self._filters[0] = self._channels_out

    for filter in self._filters:
        self._encoder.append(keras.layers.Conv2D(filters=filter,
                                                                kernel_size=5,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))
    for filter in reversed(self._filters):
        self._decoder.append(keras.layers.Conv2DTranspose(filters=filter,
                                                                kernel_size=5,
                                                                strides=1,
                                                                activation=tf.nn.leaky_relu,
                                                                padding="same"))
       
    #self._enc_dec.append(EncoderDecoder(filters=filters,strides=1))
    level=1
    for _ in range(self._n_bins):
     r = self._rows_patch * level
     c = self._columns_patch * level
     #self._upsampling.append(keras.layers.Conv2DTranspose(filters=self._channels_out+1,
#                                                kernel_size=5,
#                                                strides=level,
#                                                activation=tf.nn.leaky_relu,
#                                                padding="same"))
#
     self._upsampling.append(UpSampling2DBilinear(size=(r,c)))
     self._coord_upsampling.append(UpSampling2DBilinear(size=(r,c)))
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

    flowvar = inputs[0]
    coordinates = inputs[1]
   
    features , scores = self._scorer(flowvar) #scores shape [BS,NP]

    I     = self._ranking(scores) #indices shape [BS*NP]

    features = tf.concat([flowvar,features],axis=-1)

    patches     = self.from_image_to_patch_sequence(features) #patches shape [BS*NP,h,w,C]
    coordinates = self.from_image_to_patch_sequence(coordinates) #patches shape [BS*NP,h,w,C]

    P = []
    XZ = []
    for i , idx in enumerate(I):
     p = tf.squeeze(tf.gather(patches,idx,axis=0),axis=1)
     p = self._upsampling[i](p)
     xz = tf.squeeze(tf.gather(coordinates,idx,axis=0),axis=1)
     xz = self._coord_upsampling[i](xz)

     p = tf.concat([p,xz],axis=-1)

     #p = self._enc_dec[0](p)

     for layer in self._encoder:
      p = layer(p)

     for layer in self._decoder:
      p = layer(p)
     
     P.append(p)
     XZ.append(xz)

    return P, I, XZ 

  def train_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,:-2]
    low_res_xz = inputs[:,:,:,-2:] 

    with tf.GradientTape(persistent=True) as tape0:
        
      lr_data_loss, lr_pde_loss, hr_data_loss, hr_pde_loss = self.compute_loss(low_res_true, low_res_xz)

      beta_lr = float(lr_data_loss/lr_pde_loss)

      beta_hr = float(hr_data_loss/hr_pde_loss)

      loss = lr_data_loss + self.beta[0] * beta_lr*lr_pde_loss + self.beta[1] * hr_data_loss + beta_hr * hr_pde_loss

    lossGrad = tape0.gradient(loss, self.trainable_variables)
    del tape0

    # ---- update parameters ---- #
    self.optimizer.apply_gradients(zip(lossGrad, self.trainable_variables))

    # ---- update metrics and statistics ---- #
    # track loss and mae
    self.trainMetrics['loss'].update_state(loss)
    self.trainMetrics['lr_data_loss'].update_state(lr_data_loss)
    self.trainMetrics['lr_pde_loss'].update_state(lr_pde_loss)
    self.trainMetrics['hr_data_loss'].update_state(hr_data_loss)
    self.trainMetrics['hr_pde_loss'].update_state(hr_pde_loss)
    for key in self.trainMetrics:
      self.trainStat[key] = self.trainMetrics[key].result()
    return self.trainStat

  def compute_loss(self, low_res_true, low_res_xz):

    
    true_patches = self.from_image_to_patch_sequence(low_res_true)
    XZ = low_res_xz

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(XZ)

      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(XZ)

        predicted_patches, indices, XZ = self([low_res_true,XZ])
 
        u_pred_patches = []
        v_pred_patches = []
        p_pred_patches = []
        nu_pred_patches = []
        for p in predicted_patches:
            u_pred_patches.append(p[:,:,:,0])
            v_pred_patches.append(p[:,:,:,1])
            p_pred_patches.append(p[:,:,:,2])
            nu_pred_patches.append(p[:,:,:,3])
            

      u_grad = tape1.gradient(u_pred_patches,XZ)
      v_grad = tape1.gradient(v_pred_patches,XZ)
      p_grad = tape1.gradient(p_pred_patches,XZ)
      nu_grad = tape1.gradient(nu_pred_patches,XZ)
      del tape1

    u_grad_2 = tape2.gradient(u_grad,XZ)
    v_grad_2 = tape2.gradient(v_grad,XZ)

    del tape2

    lr_data_loss , lr_pde_loss= self.compute_lr_loss(true_patches, indices[0],predicted_patches[0],u_grad[0],v_grad[0],p_grad[0],nu_grad[0],u_grad_2[0],v_grad_2[0])

    hr_data_loss , hr_pde_loss= self.compute_hr_loss(true_patches, indices[1:],predicted_patches[1:],u_grad[1:],v_grad[1:],p_grad[1:],nu_grad[1:],u_grad_2[1:],v_grad_2[1:])


    return lr_data_loss, lr_pde_loss, hr_data_loss, hr_pde_loss

  def compute_lr_loss(self,true_patches, indices, pred_p , u_grad, v_grad, p_grad, nu_grad, u_grad_2, v_grad_2):

     true_p = tf.gather(true_patches,indices,axis=0)
     true_p = tf.reshape(true_p, shape=(true_p.shape[0],self._rows_patch,self._columns_patch,self._channels_out))

     u_pred = pred_p[:,:,:,0]
     v_pred = pred_p[:,:,:,1]
     p_pred = pred_p[:,:,:,2]
     nu_pred = pred_p[:,:,:,3]

     uMse  = tf.reduce_mean(tf.square(true_p[:,:,:,0] - u_pred))
     vMse  = tf.reduce_mean(tf.square(true_p[:,:,:,1] - v_pred))
     pMse  = tf.reduce_mean(tf.square(true_p[:,:,:,2] - p_pred))
     nuMse = tf.reduce_mean(tf.square(true_p[:,:,:,3] - nu_pred))

     lr_data_loss = (1/4)*(uMse + vMse + pMse + nuMse)
 
     ux = u_grad[:,:,:,0]
     uz = u_grad[:,:,:,1]
     vx = v_grad[:,:,:,0]
     vz = v_grad[:,:,:,1]
   
     px = p_grad[:,:,:,0]
     pz = p_grad[:,:,:,1]

     nux = nu_grad[:,:,:,0]
     nuz = nu_grad[:,:,:,1]
     
     uxx = u_grad_2[:,:,:,0]
     uzz = u_grad_2[:,:,:,1]

     vxx = v_grad_2[:,:,:,0]
     vzz = v_grad_2[:,:,:,1]

     nur= 1e-2
     lr = 1.0
     ur = 5.0
     R = nur/(lr*ur)

     cont    = ux+vz
     contMse = tf.reduce_mean(tf.square(cont))

     momx    = u_pred * ux + v_pred*uz + px - R*(2*nux*ux + nuz*(uz + vx) + (1e-2 + nu_pred)*(uxx + uzz))
     momxMse = tf.reduce_mean(tf.square(momx))
     
     momy    = u_pred * vx + v_pred*vz + pz - R*(2*nuz*vz + nux*(uz + vx) + (1e-2 + nu_pred)*(vxx + vzz))
     momyMse = tf.reduce_mean(tf.square(momy))
  
     lr_pde_loss = (1/3)*(contMse + momxMse + momyMse)

     return lr_data_loss, lr_pde_loss

  def compute_hr_loss(self,true_patches, indices, predicted_patches, u_grad, v_grad, p_grad, nu_grad, u_grad_2, v_grad_2):


     uMse = 0.0
     vMse = 0.0
     pMse = 0.0
     nuMse = 0.0
     contMse = 0.0
     momxMse = 0.0
     momyMse = 0.0
     nur= 1e-2
     lr = 1.0
     ur = 5.0
     R = nur/(lr*ur)
     conta = 0.0
     level=2

     for i,idx in enumerate(indices):

      true_p = tf.gather(true_patches,idx,axis=0)
      true_p = tf.reshape(true_p, shape=(true_p.shape[0],self._rows_patch,self._columns_patch,self._channels_out))

      if true_p.shape[0] > 0: 
       
       pred_p = predicted_patches[i]
       u_pred = pred_p[:,:,:,0]
       v_pred = pred_p[:,:,:,1]
       p_pred = pred_p[:,:,:,2]
       nu_pred = pred_p[:,:,:,3]

       ugrad = u_grad[i]
       ux = ugrad[:,:,:,0]
       uz = ugrad[:,:,:,1]
       vgrad = v_grad[i]
       vx = vgrad[:,:,:,0]
       vz = vgrad[:,:,:,1]
   
       pgrad = p_grad[i]
       px = pgrad[:,:,:,0]
       pz = pgrad[:,:,:,1]

       nugrad = nu_grad[i]
       nux = nugrad[:,:,:,0]
       nuz = nugrad[:,:,:,1]
     
       ugrad2 = u_grad_2[i]
       uxx = ugrad2[:,:,:,0]
       uzz = ugrad2[:,:,:,1]

       vgrad2 = v_grad_2[i]
       vxx = vgrad2[:,:,:,0]
       vzz = vgrad2[:,:,:,1]

       cont    = ux+vz
       contMse += tf.reduce_mean(tf.square(cont))

       momx    = u_pred * ux + v_pred*uz + px - R*(2*nux*ux + nuz*(uz + vx) + (1e-2 + nu_pred)*(uxx + uzz))
       momxMse += tf.reduce_mean(tf.square(momx))
     
       momy    = u_pred * vx + v_pred*vz + pz - R*(2*nuz*vz + nux*(uz + vx) + (1e-2 + nu_pred)*(vxx + vzz))
       momyMse += tf.reduce_mean(tf.square(momy))

       pred_p = tf.keras.layers.AveragePooling2D(pool_size=(8, level), strides=(level,level), padding="same")(pred_p)
       level = level*2
   
       conta += 1
       uMse += tf.reduce_mean(tf.square(true_p[:,:,:,0] - pred_p[:,:,:,0]))
       vMse += tf.reduce_mean(tf.square(true_p[:,:,:,1] - pred_p[:,:,:,1]))
       pMse += tf.reduce_mean(tf.square(true_p[:,:,:,2] - pred_p[:,:,:,2]))
       nuMse += tf.reduce_mean(tf.square(true_p[:,:,:,3] - pred_p[:,:,:,3]))

     hr_pde_loss = (1/(3*conta))*(contMse + momxMse + momyMse)
     hr_data_loss = (1/(4*conta))*(uMse + vMse + pMse + nuMse)

     return hr_data_loss, hr_pde_loss

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
 
    low_res_true = inputs[:,:,:,:-2]
    low_res_xz = inputs[:,:,:,-2:]

    lr_data_loss, lr_pde_loss, hr_data_loss, hr_pde_loss = self.compute_loss(low_res_true, low_res_xz)

    beta_lr = float(lr_data_loss/lr_pde_loss)

    beta_hr = float(hr_data_loss/hr_pde_loss)

    loss = lr_data_loss + self.beta[0] * beta_lr*lr_pde_loss + self.beta[1] * hr_data_loss + beta_hr * hr_pde_loss
    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['lr_data_loss'].update_state(lr_data_loss)
    self.validMetrics['lr_pde_loss'].update_state(lr_pde_loss)
    self.validMetrics['hr_data_loss'].update_state(hr_data_loss)
    self.validMetrics['hr_pde_loss'].update_state(hr_pde_loss)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


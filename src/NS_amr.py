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
    TP = []
    for i , idx in enumerate(I):
     true_p = tf.squeeze(tf.gather(patches,idx,axis=0),axis=1)
     p = self._upsampling[i](true_p)
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
     TP.append(true_p[:,:,:,:self._channels_out])

    return P, TP, I, XZ 

  def train_step(self, data):

    inputs = data[0]
 
    low_res_true = inputs[:,:,:,0:4]
    low_res_xz = inputs[:,:,:,4:6] 
    Re_nulaminar = inputs[:,:,:,6:] 

    with tf.GradientTape(persistent=True) as tape0:
        
      uMse, vMse, pMse, nuMse, contMse, momxMse, momyMse = self.compute_loss(low_res_true, low_res_xz, Re, nulaminar)

      data_loss  = (1/4)*(nuMse + uMse   + vMse + pMse)

      cont_loss = contMse
      
      beta_cont = float(data_loss/contMse)

      momx_loss = momxMse

      beta_momx = float(data_loss/momxMse)

      momy_loss = momyMse

      beta_momy = float(data_loss/momyMse)

      loss = data_loss + self.beta[0]*(beta_cont*cont_loss + beta_momx*momx_loss + beta_momy*momy_loss)

    lossGrad = tape0.gradient(loss, self.trainable_variables)
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

  def compute_loss(self, low_res_true, low_res_xz,Re,nulaminar):
    
    Re = self.from_image_to_patch_sequence(Re)
    nulaminar = self.from_image_to_patch_sequence(nulaminar)
    XZ = low_res_xz

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape2:
      tape2.watch(XZ)

      with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(XZ)

        predicted_patches, true_patches, indices, XZ = self([low_res_true,XZ])
 
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

    uMse, vMse, pMse , nuMse= self.compute_data_loss(true_patches, predicted_patches)

    contMse, momxMse, momzMse = self.compute_pde_loss(u_pred_patches,v_pred_patches,nu_pred_patches,u_grad,v_grad,p_grad,nu_grad,u_grad_2,v_grad_2,Re,nulaminar,indices)

    return uMse, vMse, pMse, nuMse, contMse, momxMse, momzMse


  def compute_data_loss(self,true_patches, predicted_patches):

    uMse = 0.0
    vMse = 0.0
    pMse = 0.0
    nuMse = 0.0
    cont = 0.0

    level=1
    for i in range(self._n_bins):

     true_p = true_patches[i]
     pred_p = predicted_patches[i]
     pred_p = tf.keras.layers.AveragePooling2D(pool_size=(level, level), strides=(level,level), padding="same")(pred_p)
     level = level*2
   
     if true_p.shape[0] > 0: 

       cont += 1
       uMse += tf.reduce_mean(tf.square(true_p[:,:,:,0] - pred_p[:,:,:,0]))
       vMse += tf.reduce_mean(tf.square(true_p[:,:,:,1] - pred_p[:,:,:,1]))
       pMse += tf.reduce_mean(tf.square(true_p[:,:,:,2] - pred_p[:,:,:,2]))
       nuMse += tf.reduce_mean(tf.square(true_p[:,:,:,3] - pred_p[:,:,:,3]))

    return (1/cont)*uMse , (1/cont)*vMse , (1/cont)*pMse , (1/cont)*nuMse

  def compute_pde_loss(self,u_pred_patches,v_pred_patches,nu_pred_patches,u_grad,v_grad,p_grad,nu_grad,u_grad_2,v_grad_2,Re,nulaminar,indices):

    contMse = 0.0
    momxMse = 0.0
    momyMse = 0.0
    cont = 0.0

    level = 1
    for i , _ in enumerate(u_grad):

     u = u_pred_patches[i]
     v = v_pred_patches[i]
     nu  = nu_pred_patches[i]

     grad = u_grad[i]
     ux = grad[:,:,:,0]
     uz = grad[:,:,:,1]

     grad = v_grad[i]
     vx = grad[:,:,:,0]
     vz = grad[:,:,:,1]

     grad = p_grad[i]
     px = grad[:,:,:,0]
     pz = grad[:,:,:,1]
     
     grad = nu_grad[i]
     nux = grad[:,:,:,0]
     nuz = grad[:,:,:,1]

     grad = u_grad_2[i]
     uxx = grad[:,:,:,0]
     uzz = grad[:,:,:,1]

     grad = v_grad_2[i]
     vxx = grad[:,:,:,0]
     vzz = grad[:,:,:,1]

     if ux.shape[0]>0:

      re = tf.squeeze(tf.gather(Re,indices[i],axis=0),axis=1)
      re = tf.keras.layers.UpSampling2D(size=level,interpolation='nearest')(re)
      re = re[:,:,:,0]
      
      nul = tf.squeeze(tf.gather(nulaminar,indices[i],axis=0),axis=1)
      nul = tf.keras.layers.UpSampling2D(size=level,interpolation='nearest')(nul)
      nul = nul[:,:,:,0]


      #continuity   
      contMse += tf.reduce_mean(tf.square(ux+vz))
      #momx
      momx = u * ux + v*uz + px - re*(2*nux*ux + nuz*(uz + vx) + (nul + nu)*(uxx + uzz))
      momxMse += tf.reduce_mean(tf.square(momx))
      #momy
      momy = u * vx + v*vz + pz - re*(2*nuz*vz + nux*(uz + vx) + (nul + nu)*(vxx + vzz))
      momyMse += tf.reduce_mean(tf.square(momy))

      cont = cont+1

     level=level*2

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
    Re = inputs[:,:,:,6:7] 
    nulaminar = inputs[:,:,:,7:] 

    uMse, vMse, pMse, nuMse, contMse, momxMse, momyMse = self.compute_loss(low_res_true, low_res_xz, Re, nulaminar)

    data_loss  = (1/4)*(nuMse + uMse   + vMse + pMse)

    cont_loss = contMse
    
    beta_cont = float(data_loss/contMse)

    momx_loss = momxMse

    beta_momx = float(data_loss/momxMse)

    momy_loss = momyMse

    beta_momy = float(data_loss/momyMse)

    loss = data_loss + self.beta[0]*(beta_cont*cont_loss + beta_momx*momx_loss + beta_momy*momy_loss)
    
    self.validMetrics['loss'].update_state(loss)
    self.validMetrics['data_loss'].update_state(data_loss)
    self.validMetrics['cont_loss'].update_state(contMse)
    self.validMetrics['momx_loss'].update_state(momx_loss)
    self.validMetrics['momy_loss'].update_state(momy_loss)
    self.validMetrics['nuMse'].update_state(nuMse)

    for key in self.validMetrics:
      self.validStat[key] = self.validMetrics[key].result()
    return self.validStat


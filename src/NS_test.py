import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_model import NSModelPinn
from tensorflow.python.ops import math_ops

class NSAmrTest(NSModelPinn):
  def __init__(self, 
               image_size = [32,128,6],
               filters = [3,16,32],
               **kwargs):

    super(NSAmrTest, self).__init__(**kwargs)

    self._r  = image_size[0]
    self._c  = image_size[1]

    self._encoder = []
    self._decoder = []
   
    self._feature_upsampling = keras.layers.UpSampling2D(size=2,interpolation='bilinear')

    for f in filters:
        self._encoder.append(keras.layers.Conv2D(filters=f,
                                                kernel_size=5,
                                                strides=1,
                                                activation=tf.nn.leaky_relu,
                                                padding="same"))


    for f in reversed(filters):

        self._decoder.append(keras.layers.Conv2DTranspose(filters=f,
                                                kernel_size=5,
                                                strides=1,
                                                activation=tf.nn.leaky_relu,
                                                padding="same"))

  def call(self, inputs):

    features = inputs[0]
    coordinates = inputs[1]
    coordinates = tf.image.resize(coordinates,
                                    size=(2*self._r,2*self._c),
                                    method='bilinear')

   
    features = self._feature_upsampling(features)
    features = tf.concat([features,coordinates],axis=-1)

    for layer in self._encoder:
      features = layer(features)


    for layer in self._decoder:
      features = layer(features)


    return features , coordinates 


  def compute_data_loss(self, high_res_pred, low_res_true):

    low_res_pred = tf.image.resize(high_res_pred,
                                    size = (self._r,self._c),
                                    method='bilinear')


    uMse = tf.reduce_mean(tf.square(low_res_pred[:,:,:,0]-low_res_true[:,:,:,0]))
    vMse = tf.reduce_mean(tf.square(low_res_pred[:,:,:,1]-low_res_true[:,:,:,1]))
    pMse = tf.reduce_mean(tf.square(low_res_pred[:,:,:,2]-low_res_true[:,:,:,2]))

    return uMse, vMse, pMse

  def compute_pde_loss(self,u_grad,v_grad):

   u_x = u_grad[:,:,:,0]
   v_z = v_grad[:,:,:,1]

   contMse = tf.reduce_mean(tf.square( (u_x+v_z) - 0))

   return contMse

  def compute_loss(self, low_res_true, low_res_xz):

    high_res_xz = low_res_xz

    with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape1:
        tape1.watch(high_res_xz)

        high_res_pred , high_res_xz = self([low_res_true,high_res_xz])
 
        high_res_pred_u = high_res_pred[:,:,:,0]
        high_res_pred_v = high_res_pred[:,:,:,1]

    u_grad = tape1.gradient(high_res_pred_u,high_res_xz)
    v_grad = tape1.gradient(high_res_pred_v,high_res_xz)

    uMse, vMse, pMse = self.compute_data_loss(high_res_pred,low_res_true)

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

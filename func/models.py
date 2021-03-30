import tensorflow as tf
from losses import *
from metrics import *
import plot


class NeuralNetwork:

 def __init__(self, input_shape):

  self.input_shape = input_shape
  self.channels = input_shape[2]
  self.arch  = None
  self.model = None
 
 def setarchitecture(self, sizefilter=(3,3), stride1=(1,1), stride2=(1,1), filter1=None, filter2=None, alpha=None,lamreg=0):

  self.setinput()
  self.conv2dlayer(nfilters=int(filter1/2), sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg), data_format="channels_last")
  self.leakyrelu(alpha=alpha)
  self.conv2dlayer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dlayer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=self.channels, sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)

 def setarchitecture_deep(self, sizefilter=(3,3), stride1=(1,1), stride2=(1,1), filter1=None, filter2=None, alpha=None,lamreg=0):

  self.setinput()
  self.conv2dlayer(nfilters=int(filter1/2), sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg), data_format="channels_last")
  self.leakyrelu(alpha=alpha)
  self.conv2dlayer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dlayer(nfilters=int(filter2/2), sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dlayer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=filter2, sizefilter=sizefilter, strides=stride2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=int(filter2/2), sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=filter1, sizefilter=sizefilter, strides=stride1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)
  self.conv2dtransposelayer(nfilters=self.channels, sizefilter=sizefilter, strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(lamreg))
  self.leakyrelu(alpha=alpha)

 def setinput(self):

  self.inputs=tf.keras.Input(shape=self.input_shape)
  self.arch = self.inputs

 def leakyrelu(self,alpha=None):
  
  layer = tf.keras.layers.LeakyReLU(alpha=alpha)
  self.arch = layer(self.arch)

 def elu(self,alpha=None):
  
  layer = tf.keras.layers.Activation(tf.keras.activations.elu)
  self.arch = layer(self.arch)

 def tanh(self):

  layer = tf.keras.layers.Activation(tf.keras.activations.tanh)
  self.arch = layer(self.arch)

 def conv2dlayer(self, nfilters=None, sizefilter=None ,strides=None,padding=None,kernel_regularizer=None, data_format=None):
   
  layer = tf.keras.layers.Conv2D(filters=nfilters,
				 kernel_size=sizefilter,
				 strides = strides,
				 padding=padding,
				 kernel_regularizer = kernel_regularizer,				
				 data_format=data_format)

  self.arch = layer(self.arch)

 def clear(self):

  tf.keras.backend.clear_session()

 def conv2dtransposelayer(self, nfilters=None, sizefilter=None ,strides=None,padding=None,kernel_regularizer=None, data_format=None):

  layer = tf.keras.layers.Conv2DTranspose(filters=nfilters,
				 kernel_size=sizefilter,
				 strides = strides,
				 padding=padding,
				 kernel_regularizer = kernel_regularizer,
				 data_format=data_format)

  self.arch = layer(self.arch)

 def create_model(self):
  
  self.model = tf.keras.Model(inputs=self.inputs, outputs=self.arch)

 def compile_model(self):
  
  self.model.compile(loss=mse_total,optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),metrics=['mse',mse_ux,mse_nut])

 def fit_model(self,
                X_train,
                Y_train,
                X_val,
                Y_val,
                batch_size=64,
                epochs=50,
                shuffle=True,
                callbacks = callbacks):
  
  self.history = self.model.fit(
            [X_train], 
            [Y_train],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=[X_val,Y_val], 
            shuffle=True, 
            callbacks=callbacks)

  return

 def plot_history(self):
  return
  

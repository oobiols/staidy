import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

class DenseLayers(keras.layers.Layer):
  def __init__(self, width=[64,64,128], act='tanh', prefix='bc',\
               reg=None,last_linear = False, **kwargs):
    super(DenseLayers, self).__init__(**kwargs)
    assert len(width) > 0
    if reg != None:
      if len(reg) == 1:
        tmp = reg[0]
        reg = np.zeros(len(width))
        reg[:] = tmp
      else:
        assert len(reg) == len(width)
    else:
      reg = np.zeros(len(width))

    self.layers = []
    # rest layers
    for i, w in enumerate(width[:-1]):
      self.layers.append(keras.layers.Dense(width[i], activation=act, \
                         kernel_regularizer=keras.regularizers.l2(reg[i]),\
                         name=prefix+repr(i)))
    if last_linear:
      self.layers.append(keras.layers.Dense(width[-1], \
                         kernel_regularizer=keras.regularizers.l2(reg[-1]),\
                         name=prefix+repr(len(width)-1)))
    else:
      self.layers.append(keras.layers.Dense(width[-1], activation=act, \
                         kernel_regularizer=keras.regularizers.l2(reg[-1]),\
                         name=prefix+repr(len(width)-1)))


  def call(self, inputs):
    bc = inputs
    for layer in self.layers:
      bc = layer(bc)
    return bc

class ConvolutionDeconvolutionLayers(keras.layers.Layer):

  def __init__(self, 
               input_shape=(64,256,6),
               filters=[16,32,128,256],\
               kernel_size=(5,5),
               activation='LeakyReLU',\
               strides=(1,1),
               reg=None,\
               last_linear = False, 
               **kwargs):

    super(ConvolutionDeconvolutionLayers, self).__init__(**kwargs)
    self.filters = filters
    assert len(self.filters) > 0
    if reg != None:
      if len(reg) == 1:
        tmp = reg[0]
        reg = np.zeros(len(self.filters))
        reg[:] = tmp
      else:
        assert len(reg) == len(self.filters)
    else:
      reg = np.zeros(len(self.filters))

    self.layers = []
    for i, f in enumerate(self.filters):
      if i == 0 :
       self.layers.append(keras.layers.Conv2D(
                                              filters=f,\
                                              kernel_size = kernel_size,\
                                              padding = "same",\
                                              strides = strides,\
                                              kernel_regularizer=keras.regularizers.l2(reg[i]),
                                              input_shape=input_shape,
					      data_format = "channels_last"
                                              )
                         )
      else:
       self.layers.append(keras.layers.Conv2D(
                                              filters=f,\
                                              kernel_size = kernel_size,\
                                              padding = "same",\
                                              strides = strides,\
                                              kernel_regularizer=keras.regularizers.l2(reg[i])
                                              )
                         )


      if activation=="LeakyReLU":
       self.layers.append(keras.layers.LeakyReLU(alpha=0.1))
      elif activation=="tanh": 
       self.layers.append(keras.layers.Activation(tf.keras.activations.tanh))

    self.filters = np.flip(self.filters) 
    n = len(self.filters)
    for i, f in enumerate(self.filters):
      
      if i == (n-1):
        self.layers.append(keras.layers.Conv2DTranspose(
                                              filters=f,\
                                              kernel_size = kernel_size,\
                                              padding = "same",\
                                              strides = strides,\
					      activation = activation,
                                              kernel_regularizer=keras.regularizers.l2(reg[i])
                                              )
                         )
      else:
        self.layers.append(keras.layers.Conv2DTranspose(
                                              filters=f,\
                                              kernel_size = kernel_size,\
                                              padding = "same",\
                                              strides = strides,\
                                              kernel_regularizer=keras.regularizers.l2(reg[i])
                                              )
                         )
        if activation=="LeakyReLU":
         self.layers.append(keras.layers.LeakyReLU(alpha=0.1))
        elif activation=="tanh": 
         self.layers.append(keras.layers.Activation(tf.keras.activations.tanh))


  def call(self, inputs):
  
    pred = inputs
    for layer in self.layers:
      pred = layer(pred)
    return pred

class DenseResidualLayers(keras.layers.Layer):
  def __init__(self, width=128, act='tanh', **kwargs):
    super(DenseResidualLayers, self).__init__(**kwargs)
    self.layer0 = keras.layers.Dense(width, activation=act)
    self.layer1 = keras.layers.Dense(width, activation=act)
    self.layer2 = keras.layers.Dense(width, activation='linear')
    self.actLayer = keras.layers.Activation('tanh')

  def call(self, inputs):
    xShortcut = self.layer0(inputs)
    x = self.layer1(xShortcut)
    x = self.layer2(x)
    x = x + xShortcut
    x = self.actLayer(x)
    return x

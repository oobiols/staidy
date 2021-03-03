import numpy as np
import tensorflow as tf

def mse_ux(y_true, y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true[:,:,:,0]-y_pred[:,:,:,0]))

def mse_uy(y_true, y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true[:,:,:,1]-y_pred[:,:,:,1]))

def mse_p(y_true, y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true[:,:,:,2]-y_pred[:,:,:,2]))

def mse_nut(y_true, y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true[:,:,:,3]-y_pred[:,:,:,3]))


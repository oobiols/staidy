import numpy as np
import tensorflow as tf

def rme(y_true, y_pred):

 diff = tf.keras.backend.abs((y_true - y_pred) / tf.keras.backend.clip(tf.keras.backend.abs(y_true),
                                        tf.keras.backend.epsilon(),
                                        None))
 return 100 * tf.keras.backend.mean(diff)

def mse_total(y_true,y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred),axis=None)

def mse_total_test():

 def loss(y_pred,y_true):
  return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
 return loss

def mse_1(y_true,y_pred):

 return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred), axis=-1)

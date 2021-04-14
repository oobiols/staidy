import sys
sys.path.append('./src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import NS_model as NSModel
from NS_dataset import *
from DataGenerator import DataGenerator
import argparse
import h5py

keras.backend.set_floatx('float32')
parser = argparse.ArgumentParser()
args = parser.parse_args()

mirrored_strategy = tf.distribute.MirroredStrategy()

path_train='./h5_datasets/train/32x128/coarse_grid.h5'
h5f = h5py.File(path_train,'r')
X_train, Y_train = np.float32(np.array(h5f.get('x'))), np.float32(np.array(h5f.get('y')))
nTr = X_train.shape[0]


path_val=path_train
h5f = h5py.File(path_val,'r')
X_val, Y_val = np.float32(np.array(h5f.get('x'))), np.float32(np.array(h5f.get('y')))
nVal = X_val.shape[0]

with mirrored_strategy.scope():
 nsNet = NSModel.NSModelSymmCNN(input_shape = (32,128,4),
                               filters=[4,16,32,256], 
                               kernel_size=(5,5), 
                               strides=(1,1),
                               reg=None,
                               lastLinear=True )
                               
 nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

# callbacks
nsCB = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, min_delta=0.01,\
                                          patience=10, min_lr=0.0001)]

batch_size=1
nsNet.fit(DataGenerator(path=path_train,batch_size=batch_size),
          validation_data=DataGenerator(path=path_val,batch_size=batch_size),\
          initial_epoch=0, epochs=10,\
          steps_per_epoch=nTr//batch_size,\
          validation_steps=nVal//batch_size,\
          verbose=2, callbacks=nsCB)


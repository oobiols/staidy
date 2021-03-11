import sys
sys.path.insert(0, './func/')

import plot
import time

from models import *
from loadDataset import *
from losses import *
from metrics import *

from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def generator(path, batch_size):

   h5f = h5py.File(path,'r')
   L =  h5f.get("x_train_dataset").shape[0]
   nbatches = int(L/batch_size)
   print(nbatches)
   idx = np.linspace(0,L-batch_size,nbatches,dtype=int)
   np.random.shuffle(idx)
   counter=0

   while True:

     first = idx[counter]
     last  = first+batch_size
     X = np.float32(h5f["x_train_dataset"][first:last])
     Y = np.float32(h5f["y_train_dataset"][first:last])
     counter +=1
     yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

     if counter>= nbatches:
      print("\nresetting counter and shuffling")
      np.random.shuffle(idx)
      counter=0


path_train='/share/crsp/lab/amowli/oobiols/datasets/train/ellipse/64x256/ellipses.h5'
h5f = h5py.File(path_train,'r')
samples =  h5f.get("x_train_dataset").shape[0]

mirrored_strategy = tf.distribute.MirroredStrategy()

height=64
width=256
channels=4

input_shape = (height,width, channels)

batch_size=64
epochs=50

path_val='/share/crsp/lab/amowli/oobiols/datasets/validation/ellipse/64x256/validation.h5'
X_val , Y_val = loadTrainDataset(path_val)
X_val, Y_val = np.float32(X_val), np.float32(Y_val)

with mirrored_strategy.scope():

 cnn = NeuralNetwork(input_shape)
 cnn.setarchitecture(sizefilter=(5,5),stride1=(1,1), stride2=(1,1),filter1=32,filter2=256,alpha=0.1,lamreg=0)
 cnn.createmodel()
 cnn.model.compile(loss=mse_total,optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=['mse',mse_ux,mse_nut])

name="Adam-1e-4-shallow"
checkpoint_filepath = './weights/'+name+'-checkpoint-{epoch:02d}'

callbacks = [ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', verbose=1,
  save_best_only=False, mode='auto', save_freq=10*samples,save_weights_only=True)]

callbacks=[]

history = cnn.model.fit(generator(path_train,batch_size),
          		steps_per_epoch=int(samples/batch_size),
          		epochs=epochs,
          		verbose=1,
          		validation_data=[X_val,Y_val], 
			shuffle=True, 
			callbacks=callbacks)

plot.history(history,name)


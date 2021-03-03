import sys
sys.path.insert(0, './func/')

import plotting
import time

from metrics import *
from models import *
from loadDataset import *
from losses import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

mirrored_strategy = tf.distribute.MirroredStrategy()


#training image size (resolution of the coarse-grid data)
height=64
width=256
channels=4

#load training dataset
train_dataset_path ='./datasets/train/coarse_grid.h5'
X_train,Y_train = loadTrainDataset(train_dataset_path)

X_train, Y_train =np.float32(X_train), np.float32(Y_train)

samples=X_train.shape[0]
 
val_dataset_path='/datasets/validation/coarse_grid.h5'
X_val , Y_val = loadTrainDataset(path)
X_val, Y_val = np.float32(X_val), np.float32(Y_val)

input_shape = (height,width, channels)

#distribute to the number of available gpus
with mirrored_strategy.scope():

 cnn = NeuralNetwork(input_shape)
 cnn.setarchitecture_deep(sizefilter=(5,5),stride1=(1,1), stride2=(1,1),filter1=32,filter2=256,alpha=0.1,lamreg=0)
 cnn.createmodel()
 cnn.model.compile(loss=mse_total,optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),metrics=['mse',mse_ux,mse_nut])

batch_sizes=[64]
epochs=50
steps = [1]
 
for batch_size in batch_sizes:
 

  name="Adam-1e-3-deep-RLROP"
  checkpoint_filepath = './weights/'+name+'-checkpoint-{epoch:02d}'
 
  callbacks = [ModelCheckpoint(filepath=checkpoint_filepath, monitor='loss', verbose=1,
    save_best_only=False, mode='auto', save_freq=10*samples,save_weights_only=True)]

  callbacks=[ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, verbose=1,
    mode='auto', min_delta=0.01, cooldown=0, min_lr=0, **kwargs
)]

  #callbacks=[]

  history = cnn.model.fit([X_train], [Y_train],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=[X_val,Y_val], shuffle=True, callbacks=callbacks)


  cnn.model.save("model"+name+".h5")
  plot.history(history,name,writing=1)
  

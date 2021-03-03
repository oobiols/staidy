import settings
import sys
sys.path.insert(0, './func/')

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

num_threads=40

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)

tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)

import loadDataset
import plot
from metrics import *
from losses import mse_total
from models import *

def generator(path, batch_size):

   h5f = h5py.File(path,'r')
   L =  h5f.get("x_train_dataset").shape[0]
   nbatches = int(L/batch_size)
   idx = np.linspace(0,L-batch_size,nbatches,dtype=int)
   np.random.shuffle(idx)
   counter=0

   while True:

     first = idx[counter]
     last  = first+batch_size
     X = np.float16(h5f["x_train_dataset"][first:last])
     Y = np.float16(h5f["y_train_dataset"][first:last])
     counter +=1
     yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

     if counter>= nbatches:
      print("\nresetting counter and shuffling")
      np.random.shuffle(idx)
      counter=0

mirrored_strategy = tf.distribute.MirroredStrategy()

############################
#### Case features #########
############################
case='airfoil'
turb=1
jCells = 1024
iCells = 1024	
channels = 4

#####################################
#### Get weights from coarse model ##
#####################################
modelPath='./models/'
#modelName ='mymodel.h5'
modelName = 'modelAdam-1e-4-deep-nogenerator-5x5.h5'
#modelName = 'model-transfer-Adam-1e-5.h5
#modelName = 'model-ITL-Adam-1e-5-512x512.h5'

#GETTING COARSE MODEL WEIGHTS
loaded_model = keras.models.load_model(modelPath+modelName, custom_objects={'mse_total':mse_total, 'mse_ux':mse_ux, 'mse_nut':mse_nut})
weights = loaded_model.get_weights()


#######################################################################################
#### Create fine model and set its initial weights to the ones from the coarse model ##
#######################################################################################

learning_rate = 1e-5

with mirrored_strategy.scope():

 input_shape = (jCells,iCells, channels)
 finecnn = NeuralNetwork(input_shape)
 finecnn.setarchitecture_deep(sizefilter=(5,5),filter1=32,filter2=256,alpha=0.1,lamreg=0)
 finecnn.createmodel()
 finecnn.model.set_weights(weights)
 finecnn.model.compile(loss=mse_total,optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=['mse',mse_ux,mse_nut])


###############################
#### Dataset for TL         ###
###############################
transferDatasetPath = '../datasets/transfer/ellipse/'+str(jCells)+'x'+str(iCells)+'/transfer.h5'
h5f = h5py.File(transferDatasetPath,'r')
train_nsamples =  h5f.get("x_train_dataset").shape[0]

###############################
#### Dataset for validation  ###
###############################
validationDatasetPath = '../datasets/validation/ellipse/'+str(jCells)+'x'+str(iCells)+'/validation.h5'
h5f = h5py.File(validationDatasetPath,'r')
val_nsamples =  h5f.get("x_train_dataset").shape[0]

batch_size = 2
epochs =5

train_steps_per_epoch = int(train_nsamples/batch_size)
val_steps_per_epoch = int(val_nsamples/batch_size)

callbacks=[]
history = finecnn.model.fit(generator(transferDatasetPath,batch_size),
	                    steps_per_epoch = train_steps_per_epoch,
                            epochs=epochs,
                            verbose=1,
                            validation_data=generator(validationDatasetPath,batch_size), 
	                    validation_steps=val_steps_per_epoch,
			    shuffle=True, 
			    callbacks=callbacks)

name ="OSTTL-Adam-1e-5-1024x1024"
finecnn.model.save("model-"+name+".h5")
plot.history(history,name)

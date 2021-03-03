import settings
import sys
sys.path.insert(0, './func/')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

import loadDataset
import plotting
from metrics import *
from losses import mse_total
from models import *

mirrored_strategy = tf.distribute.MirroredStrategy()
############################
#### Case features #########
############################
case='airfoil'
turb=1
jCells = 512
iCells = 512	
channels = 4

#####################################
#### Get weights from coarse model ##
#####################################

modelPath='./models/'
#modelName ='mymodel.h5'
modelName = 'modelAdam-1e-4-deep-nogenerator-5x5.h5'

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
datasetPath = '../datasets/transfer/ellipse/'+str(jCells)+'x'+str(iCells)+'/transfer.h5'
X_train, Y_train = loadDataset.loadTrainDataset(datasetPath)

X_train = np.float32(X_train)
Y_train = np.float32(Y_train)

val_size = int(X_train.shape[0]*0.15)
###############################
#### Dataset for validation  ###
###############################
datasetPath = '../datasets/validation/ellipse/'+str(jCells)+'x'+str(iCells)+'/validation.h5'
X_val, Y_val = loadDataset.loadTrainDataset(datasetPath)
X_val, Y_val = shuffle (X_val, Y_val)

X_val = np.float32(X_val[0:val_size])
Y_val = np.float32(Y_val[0:val_size])


batch_size = 64
epochs =5

callbacks=[]
history = finecnn.model.fit([X_train], [Y_train],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=[X_val,Y_val], shuffle=True, callbacks=callbacks)

name ="transfer-Adam-1e-5-512x512"
finecnn.model.save("model-"+name+".h5")
plotting.history(history,name)

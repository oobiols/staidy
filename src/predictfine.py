import settings
import sys
sys.path.insert(0, './func/')
import get
import h5py
import postProcess
import loadDataset
import numpy as np
from metrics import *
import tensorflow as tf
from tensorflow import keras
from losses import mse_total
from models import *

NUM_AVAIL_CORES=40
config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_AVAIL_CORES,
                        inter_op_parallelism_threads=20,
                        allow_soft_placement=True,
                        device_count = {'CPU': NUM_AVAIL_CORES})

session = tf.compat.v1.Session(config=config)


mirrored_strategy = tf.distribute.MirroredStrategy()
############################
#### Case features #########
############################
case='airfoil'
turb=1
jCells = 256
iCells = 256
channels = 4
###############################
#### CNN model for inference ##
###############################
modelPath='./models/'
modelName ='model-transfer-Adam-1e-5.h5'
#modelName = 'mymodel.h5'

#GETTING COARSE MODEL WEIGHTS
loaded_model = keras.models.load_model(modelPath+modelName, custom_objects={'mse_total':mse_total, 'mse_ux':mse_ux, 'mse_nut':mse_nut})
weights = loaded_model.get_weights()

#CREATING NEW FINE MODEL AND SETTING WEIGHTS
with mirrored_strategy.scope():

 input_shape = (jCells,iCells, channels)
 fineCNN = NeuralNetwork(input_shape)
 fineCNN.setarchitecture_deep(sizefilter=(5,5),filter1=32,filter2=256,alpha=0.1,lamreg=0)
 fineCNN.createmodel()

fineCNN.model.set_weights(weights)

###############################
#### Dataset for inference  ###
###############################
datasetPath = '../datasets/test/NACA0012/'+str(jCells)+'x'+str(iCells)+'/3degpitch.h5'
x_predict, y_true = loadDataset.loadTrainDataset(datasetPath)

x_predict = np.float32(x_predict)
y_predict = fineCNN.model.predict(x_predict, batch_size=1, verbose=1)

CNN="ellipses"
postProcess.save_nondim_predicted_fields(y_predict, int(jCells), CNN, turb)

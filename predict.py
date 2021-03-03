import settings
import sys
sys.path.insert(0, './func/')
import get
import h5py
import postProcess
import loadDataset
import matplotlib.pyplot as plt
import numpy as np
import time
import mapping
import interpolate
from metrics import *
import tensorflow as tf

#NUM_AVAIL_CORES=40
#config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_AVAIL_CORES, 
#                        inter_op_parallelism_threads=20, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU': NUM_AVAIL_CORES})

#session = tf.compat.v1.Session(config=config)


############################
#### Case features #########
############################
case='airfoil'
multinet=0
turb=1
jCells = [64]
iCells = 256
###############################
#### CNN model for inference ##
###############################
modelPath='./models/'
modelName = 'mymodel.h5'
#LOADING MODEL
from tensorflow import keras
from losses import mse_total

loaded_model = keras.models.load_model(modelPath+modelName, custom_objects={'mse_total':mse_total, 'mse_ux':mse_ux, 'mse_nut':mse_nut})
###############################
#### Dataset for inference  ###
###############################
datasetPath = '../datasets/test/NACA0012/64x256/3degpitch.h5'
print("Loading test dataset")
x_predict, y_true = loadDataset.loadTrainDataset(datasetPath)
print("predicting")
y_predict = loaded_model.predict(x_predict, batch_size=1, verbose=1)

CNN="ellipses"
print("writing")
postProcess.save_nondim_predicted_fields(y_predict, int(jCells[0]), CNN, turb)



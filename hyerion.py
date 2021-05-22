import sys
sys.path.append('./src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import NS_model as NSModel
from NS_dataset import *
from data_generator import DataGenerator, SimpleGenerator
from Dataset import Dataset
from sklearn.utils import shuffle
import argparse
import h5py
import plot
from tensorflow import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

keras.backend.set_floatx('float32')
parser = argparse.ArgumentParser()
parser.add_argument('-he', '--height', type=int, default=64,\
                    help='height of the single image')
parser.add_argument('-w', '--width', type=int, default=256,\
                    help='width of the single image')
parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,\
                    help='learning rate')
parser.add_argument('-lcont', '--lambdacont', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-lmomx', '--lambdamomx', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-lmomz', '--lambdamomz', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-e', '--epochs', type=int, default=10,\
                    help='number of epochs to train for')
parser.add_argument('-a', '--architecture', type=str, default="deep", \
                    help='shallow or deep CNN')
parser.add_argument('-ns', '--navierstokes', type=int, default=0, \
                    help='Include NS equations in loss function')
parser.add_argument('-act', '--activation', type=str, default="LeakyReLU", \
                    help='hidden layers activation function')
parser.add_argument('-bs', '--batchsize', type=int, default=64, \
                    help='global batch size')
parser.add_argument('-rlr', '--reducelr', type=int, default=0, \
                    help='include reduce lr on plateau callback')


args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

size = [args.height,args.width,4]

ds = DatasetNoWarmup(size=size,
		     add_coordinates=True)

ellipses = ["006","007","008","01","015","02","025","03","035","45"]
e="005"
pathFile = "nwellipse"+e
ds.set_name(pathFile)
X, Y =ds.load_dataset()

for i in ellipses:
 pathFile = "nwellipse"+i
 ds.set_name(pathFile)
 x , y = ds.load_dataset()
 X , Y = np.append(X,x,axis=0) , np.append(Y,y,axis=0)


if args.architecture == "deep":
 filters = [4,16,32,128]
else:
 filters = [4,32,128]


name = "arch_"+args.architecture+"_epochs_"+str(args.epochs)+"_lr_"+str(args.learningrate)+"_bs_"+str(args.batchsize)+"_act_"+args.activation+"_reg_"+str(args.lambdacont)+"_RLR_"+str(args.reducelr)

with mirrored_strategy.scope():

 if args.navierstokes==False:

  nsNet = NSModel.NSModelSymmCNN(input_shape = (args.height,args.width,4),
                               filters=filters, 
 			       activation=args.activation,
                               kernel_size=(5,5), 
                               strides=(1,1),
			       global_batch_size = args.batchsize,	
                               reg=None,
                               lastLinear=True )
  name = name+"_DataOnly"

 else:
  nsNet = NSModel.NSModelPinn(input_shape = (args.height,args.width,6),
                               filters=filters, 
				activation=args.activation,
                               kernel_size=(5,5), 
                               strides=(1,1),\
 				global_batch_size = args.batchsize,
  				beta=[args.lambdacont,args.lambdamomx, args.lambdamomz],
                               reg=None)

  name = name+"_Data+Pde"
 nsNet.build((None,64,256,4))
 nsNet.summary()
 nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learningrate))


nsCB=[]
if (args.reducelr):
 nsCB=[    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, min_delta=1e-3,\
                                          patience=25, min_lr=1e-7)]

#history = nsNet.fit(x=X_train,
#                    y=Y_train,
#                    batch_size=args.batchsize,
#                    validation_data=(X_val,Y_val),\
#                    initial_epoch=0, 
#                    epochs=args.epochs,\
#                    verbose=1, 
#              	  callbacks=nsCB,
#              	  shuffle=True)
#
#plot.history(history,name=name)

import sys
sys.path.append('./src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from NS_amr import *
from data_generator import DataGenerator, SimpleGenerator
from Dataset import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse
import h5py
import plot
from tensorflow import keras
from tensorflow.keras import mixed_precision

#mixed_precision.set_global_policy('mixed_float16')
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
parser.add_argument('-bs', '--batchsize', type=int, default=64, \
                    help='global batch size')
parser.add_argument('-ph', '--patchheight', type=int, default=32, \
                    help='mask some patches')
parser.add_argument('-pw', '--patchwidth', type=int, default=128, \
                    help='mask some patches')
parser.add_argument('-nb', '--numberbins', type=int, default=4, \
                    help='number of projection dimentions for the patch encoder')
parser.add_argument('-mn', '--modelname', type=str, default="amr", \
                    help='number of projection dimentions for the patch encoder')

args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width]
patch_size =[args.patchheight,args.patchwidth]

X = np.load('./cylinder_lr.npy')
X[:,:,:,3:] /= 500

#X, x, _, _ = train_test_split(X,X,test_size=0.1)
#
#ntrain = X.shape[0]
#nval = x.shape[0]
#
#ntrain = ntrain//args.batchsize
#nval = nval//args.batchsize

X = X[0:1000]
x = X[0:4]

name = "AMR_epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reg_"+str(args.lambdacont)+\
       "_"+args.modelname

nsNet =  NSAmrScorer(
               image_size = [args.height,args.width,5],
               patch_size = [args.patchheight,args.patchwidth],
               scorer_filters=[4,16,32],
               filters = [3,16,64],
               scorer_kernel_size = 5,
               batch_size = args.batchsize,
               nbins =args.numberbins,
    	       beta =[args.lambdacont,1.0,1.0]
               )

optimizer = keras.optimizers.Adam(learning_rate=args.learningrate)
nsNet.compile(optimizer=optimizer,
	      run_eagerly=True)

nsCB = []
history = nsNet.fit(x=X,
                    y=X,
                    batch_size=args.batchsize, 
                    validation_data=(x,x),
                    initial_epoch=0, 
                    epochs=args.epochs,\
                    verbose=1, 
               	    callbacks=nsCB,
              	    shuffle=True)

#nsNet.summary()
plot.history(history,name=name)
nsNet.save('./models/'+name)

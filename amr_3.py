import sys
sys.path.append('./src')
import os
import argparse
import plot

import numpy as np
import tensorflow as tf

from tensorflow import keras
from NS_amr_3 import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#mixed_precision.set_global_policy('mixed_float16')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

keras.backend.set_floatx('float32')
parser = argparse.ArgumentParser()
parser.add_argument('-he', '--height', type=int, default=32,\
                    help='height of the single image')
parser.add_argument('-w', '--width', type=int, default=128,\
                    help='width of the single image')
parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,\
                    help='learning rate')
parser.add_argument('-llr', '--lambdalr', type=float, default=0.1,\
                    help='constant for the pde loss')
parser.add_argument('-lhr', '--lambdahr', type=float, default=0.1,\
                    help='constant for the pde loss')
parser.add_argument('-e', '--epochs', type=int, default=2,\
                    help='number of epochs to train for')
parser.add_argument('-bs', '--batchsize', type=int, default=4, \
                    help='global batch size')
parser.add_argument('-ph', '--patchheight', type=int, default=8, \
                    help='mask some patches')
parser.add_argument('-pw', '--patchwidth', type=int, default=32, \
                    help='mask some patches')
parser.add_argument('-nb', '--numberbins', type=int, default=2, \
                    help='number of projection dimentions for the patch encoder')
parser.add_argument('-mn', '--modelname', type=str, default="amr", \
                    help='number of projection dimentions for the patch encoder')

args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width]
patch_size =[args.patchheight,args.patchwidth]

X = np.load('./datasets/channelflow_lr_turb.npy')
#Xfp = np.load('./datasets/flatplate_lr_turb.npy')

#X = np.append(X,Xfp,axis=0)
channels = X.shape[3]

X, x, _, _ = train_test_split(X,X,test_size=0.1)

ntrain = X.shape[0]
nval = x.shape[0]
ntrain = ntrain//args.batchsize
nval = nval//args.batchsize

X = X[0:24]
x = x[0:4]

name = "AMR_epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reglrdata_"+str(args.lambdalr)+\
       "_reghrpde_"+str(args.lambdahr)+\
       "_nb_"+str(args.numberbins)+\
       "_"+args.modelname

nsNet =  NSAmrScorer(
               image_size = [args.height,args.width,channels],
               patch_size = [args.patchheight,args.patchwidth],
               scorer_filters=[4,16,32],
               filters = [4,16,64],
               scorer_kernel_size = 5,
               batch_size = args.batchsize,
               nbins =args.numberbins,
    	       beta =[args.lambdalr,args.lambdahr,1.0]
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
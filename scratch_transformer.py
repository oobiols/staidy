import sys
sys.path.append('./src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from NS_transformer import *
from NS_attention import *
from NS_model import  NSModelTransformerPinn
from data_generator import DataGenerator, SimpleGenerator
from Dataset import Dataset
from sklearn.utils import shuffle
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
parser.add_argument('-rlr', '--reducelr', type=int, default=0, \
                    help='include RLRoP callback')
parser.add_argument('-pa', '--patience', type=int, default=10, \
                    help='number of patience epochs for RLRoP')
parser.add_argument('-m', '--masking', type=int, default=1, \
                    help='mask some patches')
parser.add_argument('-hp', '--patchheight', type=int, default=32, \
                    help='mask some patches')
parser.add_argument('-wp', '--patchwidth', type=int, default=128, \
                    help='mask some patches')
parser.add_argument('-ah', '--attention', type=int, default=12, \
                    help='number of attention heads')
parser.add_argument('-pr', '--projection', type=int, default=64, \
                    help='number of projection dimentions for the patch encoder')
parser.add_argument('-t', '--transformers', type=int, default=12, \
                    help='number of projection dimentions for the patch encoder')
parser.add_argument('-s' '--strides', type=int, default=2, \
                    help='number of projection dimentions for the patch encoder')
parser.add_argument('-k' '--kernelsize', type=int, default=5, \
                    help='number of projection dimentions for the patch encoder')

args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width]
patch_size =[args.patchheight,args.patchwidth]
masking=args.masking

X_train = np.load('./train_input.npy')
X_val = np.load('./validation_input.npy')

X_train= np.append(X_train,X_val,axis=0)

name = "epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reg_"+str(args.lambdacont)+\
       "_RLR_"+str(args.reducelr)+\
       "_masking_"+str(args.masking)+\
       "_projection_"+str(args.projection)+\
       "_attention_"+str(args.attention)+\
       "_Transformer"


factor = 16
f = int(np.sqrt(factor))
filters=[4,8]
nsNet =  NSAttention(image_size = [args.height,args.width,6],
                     factor = f,
                     query_dimension=100,
                     value_dimension=100,
		     filters=[4,8],
                     strides=args.strides,
                     kernel_size=args.kernelsize,
		     beta=[args.lambdacont,args.lambdamomx,args.lambdamomz])


optimizer = keras.optimizers.Adam(learning_rate=args.learningrate)
nsNet.compile(optimizer=optimizer,
	      run_eagerly=True)
nsCB=[]


if (args.reducelr):
 nsCB=[    keras.callbacks.ReduceLROnPlateau(monitor='loss',\
						 factor=0.8,\
						 min_delta=1e-3,\
      						 patience=args.patience,
						 min_lr=1e-7)
      ]

history = nsNet.fit(x=X_train,
                    y=X_train,
                    batch_size=args.batchsize,
                    validation_split=0.1,\
                    initial_epoch=0, 
                    epochs=args.epochs,\
                    verbose=1, 
               	    callbacks=nsCB,
              	    shuffle=True)

plot.history(history,name=name)
nsNet.save('./models/'+name)

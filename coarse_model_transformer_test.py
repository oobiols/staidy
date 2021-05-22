import sys
sys.path.append('./src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import NS_model as NSModel
from NS_dataset import *
from NS_transformer import *
from NS_transformer_test import * 
from data_generator import DataGenerator, SimpleGenerator
from Dataset import Dataset
from sklearn.utils import shuffle
import argparse
import h5py
import plot
from tensorflow import keras
from tensorflow.keras import mixed_precision

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#mixed_precision.set_global_policy('mixed_float16')
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)

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
args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width,6]
patch_size =[args.patchheight,args.patchwidth]
masking=args.masking

ds = Dataset(size=image_size, 
	     add_coordinates = 1)

ds.set_type("test")
ds.set_name("NACA0012")
X_train, Y_train = ds.load_dataset()
X_train , Y_train = extract_2d_patches(X_train,patch_size=patch_size,masking=0), extract_2d_patches(Y_train,patch_size=patch_size,masking=0)

name = "epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reg_"+str(args.lambdacont)+\
       "_RLR_"+str(args.reducelr)+\
       "_masking_"+str(args.masking)+\
       "_projection_"+str(args.projection)+\
       "_attention_"+str(args.attention)+\
       "_Transformer"

#with mirrored_strategy.scope():
nsNet = NSTransformer(image_size = image_size,
                      filter_size =[16,16],
	  	      sequence_length=196,
                      patch_size=[args.patchheight,args.patchwidth],
                      projection_dim_encoder=args.projection*12,
                      projection_dim_attention=args.projection,
                      num_heads=args.attention,
                      transformer_layers=args.transformers,
                      global_batch_size = args.batchsize,
	              beta=[args.lambdacont,args.lambdamomx, args.lambdamomz])

nsNet.build(input_shape=[(None,4,32,128,4),(None,4,32,128,2)])
nsNet.summary()
optimizer = keras.optimizers.Adam(learning_rate=args.learningrate)
#optimizer = mixed_precision.LossScaleOptimizer(optimizer)
#nsNet.compile(optimizer=optimizer,
#	      run_eagerly=False)

#nsNet.set_weights('./ViT/ViT-B_16.npz')

nsCB=[]
if (args.reducelr):
 nsCB=[    keras.callbacks.ReduceLROnPlateau(monitor='loss',\
						 factor=0.75,\
						 min_delta=1e-2,\
      						 patience=args.patience,
						 verbose=1,
						 min_lr=1e-7)
      ]

#print("X train shape ", X_train.shape)
#print("Y train shape ",Y_train.shape)
#print("-------")
#history = nsNet.fit(x=X_train[0:200,:,:,:,:],
#                    y=Y_train[0:200,:,:,:,:],
#                    batch_size=args.batchsize,
#                    validation_data=(X_train[0:2,:,:,:,:],Y_train[0:2,:,:,:,:]),\
#                    initial_epoch=0, 
#                    epochs=args.epochs,\
#                    verbose=1, 
#               	    callbacks=nsCB,
#              	    shuffle=True)
#
#plot.history(history,name=name)
#nsNet.save('./models/'+name)

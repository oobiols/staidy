import sys
sys.path.append('./src')
from losses import *
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import NS_model as NSModel
from NS_dataset import *
from NS_transformer import *
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
parser.add_argument('-ah', '--attention', type=int, default=4, \
                    help='number of attention heads')
parser.add_argument('-pr', '--projection', type=int, default=4, \
                    help='number of projection dimentions for the patch encoder')
args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width,6]
patch_size =[args.patchheight,args.patchwidth]
masking=args.masking

##### Validation dataset #####
###############################
ds = Dataset(size=image_size, 
	     add_coordinates = 1)

ds.set_type("validation")
ds.set_name("ellipse03")
X_val, Y_val = ds.load_dataset()
X_val , Y_val = extract_2d_patches(X_val,patch_size=patch_size,masking=0), extract_2d_patches(Y_val,patch_size=patch_size,masking=0)
X_val,Y_val = shuffle(X_val,Y_val)
print(X_val.shape)

ds.set_type("train")
ds.set_name("ellipse045")
X, Y  = ds.load_dataset()
X , Y = extract_2d_patches(X,patch_size=patch_size,masking=0), extract_2d_patches(Y,patch_size=patch_size,masking=0)
print(X.shape)

X_val =np.append(X_val,X,axis=0)
Y_val = np.append(Y_val,Y,axis=0)

print(X_val.shape)

#### Training dataset #####
###########################

ds.set_type("train")
ellipses=["025","035","055","075","008","015","006","007","01","02"]

e="005"
pathFile = "ellipse"+e
ds.set_name(pathFile)
X_train, Y_train =ds.load_dataset()
X_train , Y_train = extract_2d_patches(X_train,patch_size=patch_size,masking=args.masking), extract_2d_patches(Y_train,patch_size=patch_size, masking=0)


for i in ellipses:
 pathFile = "ellipse"+i
 ds.set_name(pathFile)
 X , Y = ds.load_dataset()
 X , Y = extract_2d_patches(X,patch_size=patch_size,masking=args.masking), extract_2d_patches(Y,patch_size=patch_size,masking=0)
 X_train , Y_train = np.append(X_train,X,axis=0) , np.append(Y_train,Y,axis=0)

name = "epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reg_"+str(args.lambdacont)+\
       "_RLR_"+str(args.reducelr)+\
       "_masking_"+str(args.masking)+\
       "_projection_"+str(args.projection)+\
       "_attention_"+str(args.attention)+\
       "_Transformer"

nsNet = NSModel.NSModelTransformerPinn(image_size = image_size,
                                       patch_size=patch_size,
                                       projection_dim=args.projection,
                                       num_heads=args.attention,
                                       transformer_layers=1,
                                       masking=args.masking,
          			       global_batch_size = args.batchsize,
				       beta=[args.lambdacont,args.lambdamomx, args.lambdamomz])
                                    #   reg=None)

nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learningrate),
	      run_eagerly=True)

nsCB=[]
if (args.reducelr):
 nsCB=[    keras.callbacks.ReduceLROnPlateau(monitor='loss',\
						 factor=0.8,\
						 min_delta=1e-3,\
      						 patience=args.patience,
						 min_lr=1e-7)
      ]

print("X train shape ", X_train.shape)
print("Y train shape ",Y_train.shape)
history = nsNet.fit(x=X_train,
                    y=Y_train,
                    batch_size=args.batchsize,
                    validation_data=(X_val,Y_val),\
                    initial_epoch=0, 
                    epochs=args.epochs,\
                    verbose=1, 
               	    callbacks=nsCB,
              	    shuffle=True)

plot.history(history,name=name)

import sys
sys.path.append('./src')
import os
from losses import *
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
parser.add_argument('-lr', '--learningrate', type=float, default=2e-4,\
                    help='learning rate')
parser.add_argument('-lcont', '--lambdacont', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-pa', '--patience', type=int, default=10, \
                    help='number of patience epochs for RLRoP')
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
parser.add_argument('-lrt', '--lrate', type=float, default=2e-4,\
                    help='learning rate')
parser.add_argument('-bs', '--batchsize', type=int, default=64, \
                    help='global batch size')
parser.add_argument('-rlr', '--reducelr', type=int, default=0, \
                    help='include reduce lr on plateau callback')


args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

size = [args.height,args.width,4]


X_train = np.load('X_test.npy')
Y_train = np.load('Y_test.npy')
X_val = X_train[0:10]
Y_val = Y_train[0:10]

if args.architecture == "deep":
 filters = [16,32,128,256]
else:
 filters = [4,32,128]


name = "arch_"+args.architecture+"_epochs_"+str(args.epochs)+"_lr_"+str(args.learningrate)+"_bs_"+str(args.batchsize)+"_act_"+args.activation+"_reg_"+str(args.lambdacont)+"_RLR_"+str(args.reducelr)+"_patience_"+str(args.patience)

with mirrored_strategy.scope():


  nsNet = NSModel.NSModelSymmCNN(input_shape = (args.height,args.width,6),
                               filters=filters, 
 			       activation=args.activation,
                               kernel_size=(5,5), 
                               strides=(1,1),
                               reg=None,
                               lastLinear=True )
  name = name+"_DataOnly"


  nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learningrate),loss=mse_total)


nsCB = [keras.callbacks.EarlyStopping(monitor="val_loss",\
					min_delta=1e-5,\
					patience = args.patience,\
					    verbose=1,\
					    mode="auto",\
					    baseline=None,\
					    restore_best_weights=False,
)]
nsCB = []

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
nsNet.save('./models/coarse_model_test')

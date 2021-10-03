import sys
sys.path.append('./src')
import os
import argparse
import plot

import numpy as np
import tensorflow as tf

from NS_amr import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model

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
parser.add_argument('-lcont', '--lambdacont', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-lmomx', '--lambdamomx', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-lmomz', '--lambdamomz', type=float, default=1.0,\
                    help='constant for the pde loss')
parser.add_argument('-e', '--epochs', type=int, default=10,\
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
parser.add_argument('-opt', '--optimizer', type=str, default="adam", \
                    help='name of the optimizer: sgd, adam, rmsprop')
parser.add_argument('-rs', '--restart', type=int, default=0, \
                    help='number of projection dimentions for the patch encoder')


args = parser.parse_args()
mirrored_strategy = tf.distribute.MirroredStrategy()

image_size = [args.height,args.width]
patch_size =[args.patchheight,args.patchwidth]

X = np.load('./datasets/channelflow_lr_turb_nondim.npy')[:1000]
Xfp = np.load('./datasets/flatplate_lr_turb_nondim.npy')[:1000]
Xe = np.load('./datasets/ellipse_lr_turb_nondim.npy')[:1000]
#Xa = np.load('./datasets/airfoil_lr_turb_nondim.npy')[0:2000]

X = np.append(X,Xfp,axis=0)
X = np.append(X,Xe,axis=0)
channels = X.shape[3]
print(X.shape)
X, x, _, _ = train_test_split(X,X,test_size=0.1,shuffle=True)

ntrain = X.shape[0]
nval = x.shape[0]
ntrain = ntrain//args.batchsize
nval = nval//args.batchsize

X = X[0:ntrain*args.batchsize]
x = x[0:nval*args.batchsize]

name = "AMR_epochs_"+str(args.epochs)+\
       "_lr_"+str(args.learningrate)+\
       "_bs_"+str(args.batchsize)+\
       "_reg_"+str(args.lambdacont)+\
       "_nb_"+str(args.numberbins)+\
       "_opt_"+str(args.optimizer)+\
       "_ph_"+str(args.patchheight)+\
       "_"+args.modelname

path = './checkpoint/'+name
if not os.path.exists(path):
    os.makedirs(path)

filepath=path+'/model'

if (args.optimizer == "adam"):
  opt = keras.optimizers.Adam(learning_rate=args.learningrate)
elif (args.optimizer == "sgd"):
  opt = keras.optimizers.SGD(learning_rate=args.learningrate,momentum=0.99)
elif (args.optimizer == "rmsprop"):
  opt = keras.optimizers.RMSprop(learning_rate=args.learningrate)

with mirrored_strategy.scope():

 nsNet =  NSAmrScorer(
          image_size = [args.height,args.width,channels],
          patch_size = [args.patchheight,args.patchwidth],
          scorer_filters=[4,8,16],
          filters = [4,8,16],
          scorer_kernel_size = 3,
          batch_size = args.batchsize//mirrored_strategy.num_replicas_in_sync,
          nbins =args.numberbins,
          beta =[args.lambdacont,1.0,1.0]
          )

 nsNet.compile(optimizer=opt,
	      run_eagerly=True)
 nsCB = [ModelCheckpoint(filepath=filepath,
			monitor='val_loss',
			verbose=0,
			save_best_only=True,
			save_weights_only=False,
			mode='auto',
			save_freq='epoch'),
	CSVLogger(path+'/history.csv')]

 history = nsNet.fit(x=X,
                    y=X,
                    batch_size=args.batchsize, 
                    validation_data=(x,x),
                    epochs=args.epochs,\
                    verbose=1, 
               	    callbacks=nsCB,
              	    shuffle=True)

#nsNet.build(input_shape=[(None,args.height,args.width,4),(None,args.height,args.width,2)])
#initial_epoch=0
#nsNet.summary()

#if args.restart == True:
#  nsNet.load_weights(filepath)
#  initial_epoch = 68

			



plot.history(history,name=name)
nsNet.save('./models/'+name)

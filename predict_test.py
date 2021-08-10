import sys
sys.path.append('./src')
import numpy as np
import tensorflow as tf
import argparse
from Dataset import *
import NS_model as NSModel
from NS_test import *
from tensorflow import keras
import matplotlib.pyplot as plt

mirrored_strategy = tf.distribute.MirroredStrategy()
keras.backend.set_floatx('float32')
parser = argparse.ArgumentParser()
parser.add_argument('-he', '--height', type=int, default=64,\
                    help='height of the single image')
parser.add_argument('-w', '--width', type=int, default=256,\
                    help='width of the single image')
parser.add_argument('-pw', '--patchwidth', type=int, default=128,\
                    help='width of the single image')
parser.add_argument('-ph', '--patchheight', type=int, default=32,\
                    help='height of the single image')
parser.add_argument('-bs', '--batchsize', type=int, default=1,\
                    help='batch size for model.predict')
parser.add_argument('-mn', '--modelname', type=str, default='name',\
                    help='batch size for model.predict')


args = parser.parse_args()

model = keras.models.load_model('./models/'+args.modelname)
model.save_weights('./weights.h5')

amr =  NSAmrTest(
               image_size = [args.height,args.width,5],
               filters=[3,16,32],
              )

amr.build(input_shape=[(None,args.height,args.width,3),(None,args.height,args.width,2)])
amr.load_weights('./weights.h5')

x = np.load('cylinder_lr.npy')[0:args.batchsize]

flowvar = x[:,:,:,0:3]
xz = x[:,:,:,3:]/500

x =[flowvar,xz]

y , _ = amr.predict(x,batch_size=args.batchsize,verbose=1)

umin = np.min(y[0,:,:,0])
umax = np.max(y[0,:,:,0])
fig, axs = plt.subplots()
im = axs.imshow(np.float32(y[0,:,:,0]),interpolation='none')
im.set_clim(umin,umax)
plt.savefig('./amr_test.png')

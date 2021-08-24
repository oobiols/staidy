import sys
sys.path.append('./src')
import numpy as np
from own_loss_function import mse_total, mse_nut, mse_ux
import tensorflow as tf
import argparse
from Dataset import *
import NS_model as NSModel
from NS_amr import *
from tensorflow import keras

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
parser.add_argument('-nb', '--numberbins', type=int, default=4,\
                    help='batch size for model.predict')


args = parser.parse_args()

model = keras.models.load_model('./models/'+args.modelname)
model.save_weights('./weights.h5')

amr =  NSAmrScorer(
               image_size = [args.height,args.width,5],
               patch_size = [args.patchheight,args.patchwidth],
               scorer_filters=[4,16,32],
               filters = [3,16,64],
               scorer_kernel_size = 5,
               batch_size = args.batchsize,
               nbins =args.numberbins
 
              )

amr.build(input_shape=[(None,args.height,args.width,4),(None,args.height,args.width,2)])
amr.load_weights('./weights.h5')

x = np.load('channelflow_lr_turb.npy')[0:1]

flowvar = x[:,:,:,:-2]
xz = x[:,:,:,-2:]

input=[flowvar,xz]
p,idx,_ = amr.predict(input,batch_size=args.batchsize,verbose=1)

for i,patch in enumerate(p):
  np.save('./saved_patches/patch_'+str(i)+'.npy',patch)
  np.save('./saved_indices/idx_'+str(i)+'.npy',idx[i]) 

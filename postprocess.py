import sys
sys.path.append('./src')

import argparse
import numpy as np
import seaborn as sn
import tensorflow as tf

from NS_amr import *
from PostProcess import PostProcessAmr
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
                    help='name of the model whose weights will be loaded for prediction')
parser.add_argument('-nb', '--numberbins', type=int, default=4,\
                    help='batch size for model.predict')
parser.add_argument('-c', '--case', type=str, default='channelflow',\
                    help='case name for prediction')
parser.add_argument('-vn', '--variablename', type=str, default="xvelocity",\
                    help='batch size for model.predict')


args = parser.parse_args()

model = keras.models.load_model('./models/'+args.modelname)
model.save_weights('./weights.h5')

channels = 6

amr =  NSAmrScorer(
               image_size = [args.height,args.width,channels],
               patch_size = [args.patchheight,args.patchwidth],
               scorer_filters=[4,16,32],
               filters = [4,16,64],
               scorer_kernel_size = 5,
               batch_size = args.batchsize,
               nbins =args.numberbins,
               )

amr.build(input_shape=[(None,args.height,args.width,4),(None,args.height,args.width,2)])
amr.load_weights('./weights.h5')

x = np.load('./datasets/'+args.case+'_lr_turb_nondim.npy')[0:1]
xx = np.load('./datasets/'+args.case+'_lr_turb.npy')[0:1]
nuref=np.max(x[:,:,:,3])
print(nuref)
flowvar = x[:,:,:,0:4]
xz = x[:,:,:,4:6]

input=[flowvar,xz]
p, _, idx,_ = amr.predict(input,batch_size=args.batchsize,verbose=1)


pp_amr = PostProcessAmr(patches=p,
			indices=idx,
			true_data=x,
			patchheight = args.patchheight,
			patchwidth = args.patchwidth,
			height=args.height,
			width=args.width,
			case_name=args.case,
			modelname=args.modelname)

pp_amr.velocity_to_foam(uref=3.5)
pp_amr.pressure_to_foam(uref=3.5)
pp_amr.nutilda_to_foam(nuref=nuref)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--modelname', type=str, default='name',\
                    help='modelname')
parser.add_argument('-nb', '--numberbins', type=int, default=4,\
                    help='batch size for model.predict')
parser.add_argument('-he', '--height', type=int, default=32,\
                    help='batch size for model.predict')
parser.add_argument('-w', '--width', type=int, default=128,\
                    help='batch size for model.predict')
parser.add_argument('-ph', '--patchheight', type=int, default=8,\
                    help='batch size for model.predict')
parser.add_argument('-pw', '--patchwidth', type=int, default=32,\
                    help='batch size for model.predict')
parser.add_argument('-vn', '--variablename', type=str, default='xvelocity',\
                    help='name of the variable to be drawn')

args = parser.parse_args()
true_data = np.load('channelflow_lr_turb.npy')[0:1]


if (args.variablename == "xvelocity"):
    v = 0
if (args.variablename == "yvelocity"):
    v = 1
if (args.variablename == "pressure"):
    v = 2
if (args.variablename == "nutilda"):
    v = 3

npx = args.height // args.patchheight
npy = args.width // args.patchwidth

np = npx*npy

loaded_patches = []
loaded_indices = []

for i in range(args.numberbins):

    loaded_patches.append(np.load('saved_patches/patch_'+str(i)+'.npy'))
    loaded_indices.append(np.load('saved_indices/idx_'+str(i)+'.npy'))
    print(loaded_indices[i])


fig, axs = plt.subplots(npx, npy , gridspec_kw = {'wspace':0, 'hspace':-0.9})

for i in range(np):
 for j , indices in enumerate(loaded_indices):

     if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = loaded_patches[j]
         patch = patches[idx,:,:,:]

  x = i//npx
  y = i%npy

  data = patch[:,:,v]
  axs[x,y].set_xticks([])
  axs[x,y].set_yticks([])    
  im = axs[x,y].imshow(np.float32(data),interpolation='none')
  im.set_clim(umin,umax)
  

directory_name = './amr_fields/'+args.modelname
file_name = args.variablename

if not os.path.exists(directory_name):
    os.makedirs(directory_name)

plt.savefig(directory_name+'/'+file_name)
plt.close()
  
  

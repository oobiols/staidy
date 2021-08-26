import os
import argparse

import numpy as np
import seaborn as sn

import matplotlib.pyplot as plt

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
true_data = np.load('./datasets/channelflow_lr_turb.npy')[0:1]

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

npatches = npx*npy

loaded_patches = []
loaded_indices = []

for i in range(args.numberbins):
    p = np.load('./saved_patches/patch_'+str(i)+'.npy')
    loaded_patches.append(p)
    idx = np.load('./saved_indices/idx_'+str(i)+'.npy')
    if v == 0:
     print(idx)
    loaded_indices.append(idx)


fig, axs = plt.subplots(npx, npy , gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(6,2))
fig.suptitle(args.variablename+"_amr", fontsize=10)

for i in range(npatches):
 for j , indices in enumerate(loaded_indices):

     if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = loaded_patches[j]
         patch = patches[idx,:,:,:]

 x = i//npx
 y = i%npy

 data = patch[:,:,v]
 umin = np.min(true_data[0,:,:,v])
 umax = np.max(true_data[0,:,:,v])
 axs[x,y].set_xticks([])
 axs[x,y].set_yticks([])    
 axs[x,y].patch.set_edgecolor('black')  
 axs[x,y].patch.set_linewidth('1')
 hm = sn.heatmap(data, vmin = umin, vmax= umax, ax=axs[x,y],cbar=False, xticklabels=False,yticklabels=False)
 
directory_name = './amr_fields/'+args.modelname
file_name = args.variablename

if not os.path.exists(directory_name):
    os.makedirs(directory_name)

plt.savefig(directory_name+'/'+file_name,dpi=600)
plt.close()
  
plt.figure(figsize=(8,2))
data= true_data[0,:,:,v]
hm = sn.heatmap(data, vmin = umin, vmax= umax, xticklabels=False,yticklabels=False)
plt.title(args.variablename+"_true")
plt.savefig(directory_name+'/'+file_name+'_true',dpi=600)
plt.close()

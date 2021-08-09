import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_patches(image,r,c):

   patches = tf.image.extract_patches(image, sizes=(1,r,c,1), strides=(1,r,c,1),rates=(1,1,1,1),padding="SAME")

   return patches

true_data = np.load('cylinder_lr.npy')[0:1]

umax = np.max(true_data[0,:,:,0])
umin = np.min(true_data[0,:,:,0])
print(umax)
print(umin)

p0 = np.load('saved_patches/patch_0.npy')
p1 = np.load('saved_patches/patch_1.npy')
p2 = np.load('saved_patches/patch_2.npy')
p3 = np.load('saved_patches/patch_3.npy')


i0 = np.load('saved_indices/idx_0.npy')[:,0]
i1 = np.load('saved_indices/idx_1.npy')[:,0]
i2 = np.load('saved_indices/idx_2.npy')[:,0]
i3 = np.load('saved_indices/idx_3.npy')[:,0]

fig, axs = plt.subplots(4, 4 , gridspec_kw = {'wspace':0, 'hspace':-0.9})

for i in range(16):

 if i in i0:
  idx = np.where(i0==i)
  idx=idx[0][0]
  patch = p0[idx,:,:,:]
  
 elif i in i1:
  idx = np.where(i1==i)
  idx=idx[0][0]
  patch = p1[idx,:,:,:]

 elif i in i2:
  idx = np.where(i2==i)
  idx=idx[0][0]
  patch = p2[idx,:,:,:]
  
 elif i in i3:
  idx = np.where(i3==i)
  idx=idx[0][0]
  patch = p3[idx,:,:,:]
  
 x = i//4
 y = i%4
  
 data = patch[:,:,0]
 axs[x,y].set_xticks([])
 axs[x,y].set_yticks([])    
 im = axs[x,y].imshow(np.float32(data),interpolation='none')
 im.set_clim(umin,umax)
  
plt.savefig('first_patch_attempt.png')
plt.close()
  
  
##fig, axs = plt.subplots(4, 4 , gridspec_kw = {'wspace':0, 'hspace':-0.9})
###plt.subplots_adjust(wspace=0,hspace=-0.1)
##for i in range(ux_patches.shape[1]):
## for j in range(ux_patches.shape[2]):
##
##    data = ux_patches[0,i,j,:]
##    data = data.reshape((8,32))
##    axs[i,j].set_xticks([])
##    axs[i,j].set_yticks([])    
##    im = axs[i,j].imshow(np.float32(data),interpolation='none')
##    im.set_clim(umin,umax)
##    #plt.colorbar(im, ax = axs[i,j])
##
###plt.subplots_adjust(wspace=0, hspace=0)
##plt.savefig('./test.png')
##plt.close()
##
##fig, axs = plt.subplots()
##im = axs.imshow(np.float32(ux[0,:,:,0]),interpolation='none')
##im.set_clim(umin,umax)
##plt.savefig('./test_2.png')




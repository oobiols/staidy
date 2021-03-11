import sys
sys.path.insert(0, './func')

import read
import get
import os
import numpy as np
import h5py

from sklearn.utils import shuffle


#sample size in current dataset
height = 64
width  = 256
channels_input = 4
channels_output = 4

#refrence viscosity
visc = 1e-4

#1 for turbulent regime, 0 for laminar regime
turb = 1

shape = (None,height,width,channels_input)

#path where to save the dataset
save_path='./datasets/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#name of the dataset
name="coarse_grid_dataset"

file_path=path+name+'.h5'

benchmark  = "ellipse"
grid       = "ellipse"

#of one geometry, we have several configurations (different attack and pitch angles)
#nconf stands for configuration number
conf_n=1
conf_end=9
count = 0
while (nconf !=conf_end) :

    case = "case_"+str(conf_n)
    data = "train_data"
    dim = np.array([height,width,0.0, float(visc)])

    train_x_addrs, train_y_addr = read.addrs(data, "ellipse", case, grid)

    coordinates =0

    train_x=[] 
    train_y=[]
    get.case_data(train_x_addrs, train_y_addr, coordinates, dim, grid, turb, train_x,train_y)

    train_x=np.float32(np.asarray(train_x))
    train_y=np.float32(np.asarray(train_y))

    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

   #If dataset does not exist, create it
   if count==0:

    h5f.create_dataset('x_train_dataset', data = train_x,compression="lzf",chunks=True, maxshape = shape)
    h5f.create_dataset('y_train_dataset', data = train_y,compression="lzf", chunks=True, maxshape = shape)
    h5f.close() 

   #if it exists, augment it
   else:

    with h5py.File(pathFile, 'a') as hf:

      hf["x_train_dataset"].resize((hf["x_train_dataset"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x_train_dataset"][-train_x.shape[0]:] = train_x

      hf["y_train_dataset"].resize((hf["y_train_dataset"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y_train_dataset"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  


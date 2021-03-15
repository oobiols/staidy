import sys
sys.path.insert(0, './func')

import os
import h5py
import glob
import numpy as np

import get
from settings import natural_keys

class Dataset():

 def __init__ (self,size= [32,128,4]):

  self.height = size[0]
  self.width = size[1]
  self.channels = size[2]
  self.shape = (None, self.height, self.width. self.channels)


 def set_info_path(self,
                   path='./datasets/', 
                   ds_type="train",
                   ds_name="coarse_grid"):

  self.ds_type = ds_type
  self.data_source = self.ds_type+"_data"
  self.path_save = path+ds_type+'/'+str(self.height)+'x'+str(self.width)+'/'+name+'.h5'

  if not os.path.exists(self.path_save):
    os.makedirs(self.path_save)


 def create_dataset(self
                    benchmark="ellipse",
                    grid="ellipse",
                    first_case=1
                    last_case=2):
 
  
  case_number = first_case
  case_end    = last_case
  count = 0
  visc  = 1e-4
  turb  = 1

  while (case_number !=case_end) :

    case = "case_"+str(case_number)
    print("case number is ", case)
    dim = np.array([height,width,0.0, float(visc)])
    train_x_addrs, train_y_addr = self.read_addrs(self.data_source, benchmark, case, grid)
    coordinates =0
  
    train_x = [] 
    train_y = []
    get.case_data(train_x_addrs, train_y_addr, coordinates, dim, grid, turb, train_x,train_y)
    train_x=np.float32(np.asarray(train_x))
    train_y=np.float32(np.asarray(train_y))
  
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(self.path_save,"w")
     h5f.create_dataset('x_train_dataset', data = train_x,compression="lzf",chunks=True, maxshape = shape)
     h5f.create_dataset('y_train_dataset', data = train_y,compression="lzf", chunks=True, maxshape = shape)
     h5f.close() 

    else:

     with h5py.File(self.path_save, 'a') as hf:

      hf["x_train_dataset"].resize((hf["x_train_dataset"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x_train_dataset"][-train_x.shape[0]:] = train_x

      hf["y_train_dataset"].resize((hf["y_train_dataset"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y_train_dataset"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  

 def read_addrs(self,
                data, 
                benchmark, 
                case, 
                grid):
 
  x_addrs = []
  y_addr = []
 
  train_x_path   = "./" + data + "/" + case + "/input/*"
  train_x_addrs  = sorted(glob.glob(train_x_path))
  train_x_addrs  = list(train_x_addrs)
  train_x_addrs.sort(key=natural_keys)
  train_x_addrs  = train_x_addrs[0:700]
  x_addrs.append(train_x_addrs)
 
  train_y_path   = "./" + data + "/" + case + "/output/*"
  train_y_addr  = sorted(glob.glob(train_y_path))
  train_y_addr  = list(train_y_addr)
  train_y_addr.sort(key=natural_keys)
  y_addr.append(train_y_addr)
 
  x_addrs = np.asarray(x_addrs)
  y_addr  = np.asarray(y_addr)
 
  return x_addrs, y_addr

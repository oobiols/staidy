import os
import h5py
import glob
import numpy as np

import get
from settings import natural_keys

class Dataset():

 '''

 In this class, we can:

 - Create datasets in .h5 format from simulation data in h5_datasets directory
 - Load .h5 files to train/validate/test your model

 This class needs simulation data.
 This class reads OpenFOAM files stored in, 
 for example, train_data/case_1 

'''

 def __init__ (self,size= [32,128,4]):

  #Each image in the dataset is of size 'size'
  self.height = size[0]
  self.width = size[1]
  self.channels = size[2]
  self.directory = './h5_datasets/'
  self.dataset_name = 'coarse_grid'
  self.dataset_type = 'train'
  self.shape = (None, self.height, self.width, self.channels)
  self.path= self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'
  
 #defining directory and name where to save the h5 dataset

 def set_directory(self,
                   path='./h5_datasets/'):
  self.directory = path
  return 0

 def set_name(self,
		dataset_name='coarse_grid'):
  
  self.dataset_name = dataset_name
  return 0

 def set_type(self,
		dataset_type='train'):
  
  self.dataset_type = dataset_type
  return 0

 def load_dataset(self):

  self.file_path = self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'+self.dataset_name+'.h5'
  h5f = h5py.File(self.file_path,"r")

  x = h5f.get('x')
  x = np.float32(np.asarray(x))
  y = h5f.get('y')
  y = np.float32(np.asarray(y))

  return (x,y)


 def create_dataset(self,
                    benchmark="ellipse",
                    grid="ellipse",
                    first_case=1,
                    last_case=2):
 
  self.directory_path =  self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'
  if not os.path.exists(self.directory_path):
    os.makedirs(self.directory_path)
  
  case_number = first_case
  case_end    = last_case + 1
  turb  = isturb

  count = 0
  while (case_number !=case_end) :

    case = "case_"+str(case_number)
    print("case number is ", case)
    dim = np.array([self.height,self.width,0.0, float(visc)])

    #Reads the path addresses where the OpenFOAM data is stored
    train_x_addrs, train_y_addr = self.read_addrs(self.dataset_type, benchmark, case, grid)

    train_x = [] 
    train_y = []
    #Reads the primary variable values at each iteration of the OpenFOAM data and t 
    get.case_data(train_x_addrs, train_y_addr, coordinates, dim, grid, turb, train_x,train_y)
    train_x=np.float32(np.asarray(train_x))
    train_y=np.float32(np.asarray(train_y))
  
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(self.path_save+self.ds_name+'.h5',"w")
     h5f.create_dataset('x_train_dataset', data = train_x,compression="lzf",chunks=True, maxshape = self.shape)
     h5f.create_dataset('y_train_dataset', data = train_y,compression="lzf", chunks=True, maxshape = self.shape)
     h5f.close() 

    else:

     with h5py.File(self.path_save+self.ds_name+'.h5', 'a') as hf:

      hf["x_train_dataset"].resize((hf["x_train_dataset"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x_train_dataset"][-train_x.shape[0]:] = train_x

      hf["y_train_dataset"].resize((hf["y_train_dataset"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y_train_dataset"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  

 def __read_addrs(self,
                data, 
                benchmark, 
                case, 
                grid):
 
  x_addrs = []
  y_addr = []
 
  train_x_path   = "./" + data+"_data" + "/" + case + "/input/*"
  train_x_addrs  = sorted(glob.glob(train_x_path))
  train_x_addrs  = list(train_x_addrs)
  train_x_addrs.sort(key=natural_keys)
  x_addrs.append(train_x_addrs)
 
  train_y_path   = "./" + data + "/" + case + "/output/*"
  train_y_addr  = sorted(glob.glob(train_y_path))
  train_y_addr  = list(train_y_addr)
  train_y_addr.sort(key=natural_keys)
  y_addr.append(train_y_addr)
 
  x_addrs = np.asarray(x_addrs)
  y_addr  = np.asarray(y_addr)
 
  return x_addrs, y_addr

 def __case_data       (x_addrs, y_addr, coordinates, dim, grid, turb, x_train, y_train):

  
  x_addrs = x_addrs[0]
  n = len(x_addrs)

  y_interior_addr = y_addr[0]
  y_interior_addr = y_interior_addr[0]
  y_top_addr = y_interior_addr
  y_bottom_addr = y_interior_addr

  for i in range(0,n):

    x_interior_addr = x_addrs[i]
    x_bottom_addr   = []
    x_top_addr      = []

    pos = "input"
    data_cell  = single_sample(grid,     x_interior_addr, 
                              x_bottom_addr, x_top_addr,
			      dim, turb, pos)
    
    x_train.append(data_cell)

    pos = "output"
    data_cell = single_sample(grid,     y_interior_addr,  
		              y_bottom_addr, y_top_addr,
			      dim, turb, pos)


    y_train.append(data_cell)

  
  return


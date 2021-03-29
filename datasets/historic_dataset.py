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

 def __init__ (self,
              domain_size= [32,128,4],
              is_turb=1):

  #Each image in the dataset is of size 'size'
  self.height = size[0]
  self.width = size[1]
  self.channels = size[2]
  self.is_turb = is_turb
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

  count = 0
  while (case_number !=case_end) :

    case = "case_"+str(case_number)
    print("case number is ", case)

    #Reads the path addresses where the OpenFOAM data is stored
    train_x_addrs, train_y_addr = self.__read_addrs(self.dataset_type, case)

    #Reads the primary variable values at each iteration of the OpenFOAM data and t 
    train_x, train_y = self.__case_data(train_x_addrs, train_y_addr, grid, turb)
  
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(self.path_save+self.ds_name+'.h5',"w")
     h5f.create_dataset('x_train_dataset', data = train_x,compression="lzf",chunks=True, maxshape = self.shape)
     h5f.create_dataset('y_train_dataset', data = train_y,compression="lzf", chunks=True, maxshape = self.shape)
     h5f.close() 

    else:

     with h5py.File(self.path_save+self.ds_name+'.h5', 'a') as hf:

      hf["x_train_dataset"].resize((hf["x"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x_train_dataset"][-train_x.shape[0]:] = train_x

      hf["y_train_dataset"].resize((hf["y"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y_train_dataset"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  

  return 

 def __read_addrs(self,
                data,  
                case):
 
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

 def __case_data(x_addrs, 
                 y_addr, 
                 grid, 
                 turb):

  x_train = []
  y_train = []
  x_addrs = x_addrs[0]
  n = len(x_addrs)

  y_addr = y_addr[0]
  y_addr = y_addr[0]

  for i in range(0,n):

    x_addr = x_addrs[i]

    pos = "input"
    data_cell  = self.__single_sample(grid, x_addr, pos)
    
    x_train.append(data_cell)
 
    pos = "output"
    data_cell = self.__single_sample(grid,y_addr,pos)

    y_train.append(data_cell)

  
  return x_train, y_train


 def __single_sample(grid,
                     addr,
                     pos):
  


  height = self.height

  Ux, Uy, p, nuTilda = self.__get_domain(addr, self.isturb)

#  Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(bottom_addr, turb)
#  Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(top_addr, turb)

 # Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(interior_addr, turb)
 # Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(interior_addr, turb)

  Ux = self.__mapping_domain(Ux, grid)
  Uy = self.__mapping_domain(Uy, grid)
  p  = self.__mapping_domain(p,  grid)

  Ux = Ux_interior
  Uy = Uy_interior
  p  = p_interior

#  if pos == "input":

#   boundary="bottom"
#   Ux_bottom = mapping.boundary(Ux_bottom, dim, grid, Ux_interior, boundary)
#   Uy_bottom = mapping.boundary(Uy_bottom, dim, grid, Uy_interior, boundary)
#   p_bottom  = mapping.boundary(p_bottom,  dim, grid, p_interior, boundary)

#   boundary="top"
#   Ux_top = mapping.boundary(Ux_top, dim, grid, Ux_interior, boundary)
#   Uy_top = mapping.boundary(Uy_top, dim, grid, Uy_interior, boundary)
#   p_top  = mapping.boundary(p_top,  dim, grid, p_interior, boundary)

  if (turb):

   nuTilda_interior = mapping.interior(nuTilda_interior, dim, grid)
   nuTilda = nuTilda_interior

#   if pos == "input":
#    boundary="bottom"
# #   nuTilda_bottom = mapping.boundary(nuTilda_bottom, dim, grid,nuTilda_interior,boundary)
#    boundary="top"
#  #  nuTilda_top = mapping.boundary(nuTilda_top, dim, grid, nuTilda_interior,boundary)
#
#  if pos == "input":
#
# #  Ux = np.append(Ux_bottom, Ux_interior, axis = 0)
# #  Uy = np.append(Uy_bottom, Uy_interior, axis = 0)
# #  p  = np.append(p_bottom, p_interior,   axis = 0) 

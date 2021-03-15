import os
import numpy as np
import h5py


class Dataset():

 def __init__ (self,size= [32,128,4],path='./',dataset_type='train',name='coarse_grid'):

  self.height = size[0]
  self.width = size[1]
  self.channels = size[2]
  self.dataset_type = dataset_type
  self.name = name 

  self.shape = (None, self.height, self.width. self.channels)

  self.path_save=path+dataset_type+'/'+str(self.height)+ 'x'+str(self.width)+'/'+name+'.h5'
  if not os.path.exists(self.path_save):
    os.makedirs(self.path_save)

  self.source = self.dataset_type+'_data'
  self.nondim_visc = 1e-4
  self.grid = "ellipse"
  self.benchmark = "ellipse"
 
 

grids           = np.array(["ellipse"])
num_benchmarks  = canonical_flows.shape[0]
for b in range(0,num_benchmarks):
  benchmark  = canonical_flows[b,0] 
  grid       = grids[b]
  print(benchmark)
  case_number=case_start
  count = 0
  while (case_number !=case_end) :
    visc = 1e-4
    data = "train_data"
    case = "case_"+str(case_number)
    print("case number is ", case)

    turb = 1
    dim = np.array([height,width,0.0, float(visc)])
    train_x_addrs, train_y_addr = read.addrs(data, benchmark, case, grid)
    coordinates =0
    train_x=[] 
    train_y=[]
    get.case_data(train_x_addrs, train_y_addr, coordinates, dim, grid, turb, train_x,train_y)
    train_x=np.float32(np.asarray(train_x))
    train_y=np.float32(np.asarray(train_y))
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(pathFile,"w")
     h5f.create_dataset('x_train_dataset', data = train_x,compression="lzf",chunks=True, maxshape = shape)
     h5f.create_dataset('y_train_dataset', data = train_y,compression="lzf", chunks=True, maxshape = shape)
     h5f.close() 

    else:

     with h5py.File(pathFile, 'a') as hf:

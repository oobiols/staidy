import sys
sys.path.insert(0, './func')

import read
import get
import os
import numpy as np
import h5py

from sklearn.utils import shuffle

height = 64
width  = 256
channels_input = 4
channels_output = 4

case_start = 1
case_end=10


shape = (None,height,width,channels_input)

typeof="train"

path='/pub/oobiols/datasets/'+typeof+'/ellipse/'+str(height)+'x'+str(width)+'/'
if not os.path.exists(path):
    os.makedirs(path)

pathFile=path+'ellipse035.h5'

canonical_flows = np.array(   [
["ellipse"]
                              ])


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
    case = "case_"+str(case_number)
    print("case number is ", case)
    data = "train_data"
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

      hf["x_train_dataset"].resize((hf["x_train_dataset"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x_train_dataset"][-train_x.shape[0]:] = train_x

      hf["y_train_dataset"].resize((hf["y_train_dataset"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y_train_dataset"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  


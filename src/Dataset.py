import os
import h5py
import glob
import openfoamparser as Ofpp
import numpy as np

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
              size= [32,128,4],
              grid = "ellipse",
              is_turb=1,
              add_coordinates = 0):

  self.height       = size[0]
  self.width        = size[1]
  self.channels     = size[2]
  self.is_turb      = is_turb
  self.add_coordinates = add_coordinates
  self.directory    = './h5_datasets/'
  self.dataset_name = 'coarse_grid'
  self.dataset_type = 'train'
  self.grid         = grid
  self.path         = self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'
  self.shape        = [None,self.height,self.width,self.channels]
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
  x = np.asarray(x)
  y = h5f.get('y')
  y = np.asarray(y[:,:,:,0:4])

  x[:,:,:,3] = 1e-3*x[:,:,:,3]/1e-4
  y[:,:,:,3] = 1e-3*y[:,:,:,3]/1e-4

  if self.add_coordinates==False:
   x = x[:,:,:,0:4]
   y = y[:,:,:,0:4]
   
  return (x,y)


 def join_datasets(self, path_1,path_2):

  h5f = h5py.File(path_1,"r")

  x , y = h5f.get('x'), h5f.get('y')
  x_1 , y_1 = np.float16(np.asarray(x)) , np.float16(np.asarray(y))
 
  h5f.close()

  h5f = h5py.File(path_2,"r")

  x , y = h5f.get('x'), h5f.get('y')
  x , y = np.float16(np.asarray(x)) , np.float16(np.asarray(y))

  h5f.close()

  x = np.append(x_1,x,axis=0)
  y = np.append(y_1,y,axis=0)

  h5f = h5py.File(save_path+'joined_dataset.h5',"w")
  h5f.create_dataset('x', data = x,compression="lzf",chunks=True, maxshape = self.shape)
  h5f.create_dataset('y', data = y,compression="lzf", chunks=True, maxshape = self.shape)
  h5f.close() 

 def create_dataset(self,
                    first_case=1,
                    last_case=2):
 
  self.directory_path =  self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'
  self.file_path =  self.directory_path + self.dataset_name + '.h5'
  if not os.path.exists(self.directory_path):
    os.makedirs(self.directory_path)
  
  case_number = first_case
  case_end    = last_case + 1

  count = 0

  while (case_number !=case_end) :

    case = "case_"+str(case_number)
    print("case number is ", case)

    if self.add_coordinates:
     self.xyz = self.__get_coordinates(self.dataset_type,case)
    #Reads the path addresses where the OpenFOAM data is stored
    train_x_addrs, train_y_addr = self.__read_addrs(self.dataset_type, case)
    #Reads the primary variable values at each iteration of the OpenFOAM data and t 
    train_x, train_y = self.__case_data(train_x_addrs, train_y_addr)
  
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(self.file_path,"w")
     h5f.create_dataset('x', data = train_x,compression="gzip",compression_opts=5,chunks=True, maxshape = self.shape)
     h5f.create_dataset('y', data = train_y,compression="gzip",compression_opts=5, chunks=True, maxshape = self.shape)
     h5f.close() 

    else:

     with h5py.File(self.file_path, 'a') as hf:

      hf["x"].resize((hf["x"].shape[0] + train_x.shape[0]), axis = 0)
      hf["x"][-train_x.shape[0]:] = train_x

      hf["y"].resize((hf["y"].shape[0] + train_y.shape[0]), axis = 0)
      hf["y"][-train_y.shape[0]:] = train_y
       
      hf.close()

    case_number=case_number+1
    count=count+1  

  return 


 def __get_coordinates(self,
			data,
			case):

  
  coord_path   = "./" + data+"_data" + "/" + case + "/"
  xyz = np.loadtxt(coord_path+"xyz.txt")

  return xyz


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
 
  train_y_path   = "./" + data+"_data" + "/" + case + "/output/*"
  train_y_addr  = sorted(glob.glob(train_y_path))
  train_y_addr  = list(train_y_addr)
  train_y_addr.sort(key=natural_keys)
  y_addr.append(train_y_addr)
 
  x_addrs = np.asarray(x_addrs)
  y_addr  = np.asarray(y_addr)
 
  return x_addrs, y_addr

 def __case_data(self,
                 x_addrs, 
                 y_addr):

  x_train = []
  y_train = []
  x_addrs = x_addrs[0]
  n = len(x_addrs)

  y_addr = y_addr[0]
  y_addr = y_addr[0]

  for i in range(0,n):

    x_addr = x_addrs[i]

    data_cell  = self.__single_sample(x_addr)
    
    x_train.append(data_cell)
 
    data_cell = self.__single_sample(y_addr)

    y_train.append(data_cell)

  
  return np.float16(np.asarray(x_train)), np.float16(np.asarray(y_train))


 def __single_sample(self,
                     addr):
  

  Ux, Uy, p, nuTilda = self.__get_domain(addr)

  Ux = self.__map_domain(Ux)
  Uy = self.__map_domain(Uy)
  p  = self.__map_domain(p)

  Ux = Ux.reshape([Ux.shape[0], Ux.shape[1], 1])
  Uy = Uy.reshape([Uy.shape[0], Uy.shape[1], 1])
  p =  p.reshape( [p.shape[0],  p.shape[1],  1])

  if (self.grid == "channel_flow"):

    height = self.height
    Ux_avg = Ux[int(height/2),0,0]
    Uy_avg = Uy[int(height/2),0,0]
    uavg = Ux_avg
    nuTildaAvg = 1e-4

  elif (self.grid == "ellipse"):

    uavg = 0.6
    nuTildaAvg = 1e-3

  Ux /= uavg
  Uy /= uavg
  p /= uavg*uavg

  data    = np.concatenate( (Ux, Uy) , axis=2)  
  data    = np.concatenate( (data, p), axis=2)

  if (self.is_turb):
   nuTilda = self.__map_domain(nuTilda)
   nuTilda = nuTilda.reshape([nuTilda.shape[0], nuTilda.shape[1], 1])
   nuTilda /= nuTildaAvg
   data   = np.concatenate( ( data, nuTilda), axis=2) 

  if (self.add_coordinates):
    x,z = self.xyz[:,0], self.xyz[:,2]
    x,z = self.__map_domain(x), self.__map_domain(z)
    x = x.reshape([x.shape[0], x.shape[1],1])
    z = z.reshape([z.shape[0], z.shape[1],1])

    data = np.concatenate ( (data,x), axis=2)  
    data = np.concatenate ( (data,z), axis=2)


  return np.float16(data)

 def __get_domain(self,
                    addr):

   U = np.float16(Ofpp.parse_internal_field(addr+"/U"))
   p = np.float16(Ofpp.parse_internal_field(addr+"/p"))

   if (self.is_turb):
    nuTilda     = np.float16(Ofpp.parse_internal_field(addr+"/nuTilda"))
   else:
    nuTilda = 0

   Ux          = U[:,0]
   Uy          = U[:,2]

   return Ux, Uy, p, nuTilda


 def __map_domain(self, arr):

  if self.grid=="ellipse":
     
    height = self.height
    width  = self.width

    w = int(width/4)

    arr = arr.reshape( [height, width] )

    b_1 = np.empty([0, w], float)
    b_2 = np.empty([0, w], float)
    b_3 = np.empty([0, w], float)
    b_4 = np.empty([0, w], float)

    for i in range(0,int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_1  = np.append(b_1, line, axis=0)

    for i in range(int(height/4),2*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_2  = np.append(b_2, line, axis=0)
 
    for i in range(2*int(height/4),3*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_3  = np.append(b_3, line, axis=0)

    for i in range(3*int(height/4), 4*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_4  = np.append(b_4, line, axis=0)

    b_3 = np.flip(b_3, axis=1)
    b_4 = np.flip(b_4, axis=1)
 
    ret = np.append(b_3,b_4, axis=1)
    ret = np.append(ret,b_2, axis=1)
    ret = np.append(ret,b_1, axis=1)

  elif (self.grid == "channel_flow"):

    ret = arr.reshape( [self.height, self.width] )

  return ret


class DatasetNoWarmup(Dataset):
  def __init__(self):
    super(DatasetNoWarmup, self).__init__(**kwargs)
    self.add_coordinates=True

  def __case_data(self,
                 x_addrs, 
                 y_addr):

    x_train = []
    y_train = []
  
    x_addr = x_addrs[0]
    y_addr = y_addr[0]
  
    data_cell  = self.__single_sample(x_addr,"input")
    
    x_train.append(data_cell)
  
    data_cell = self.__single_sample(y_addr,"output")
  
    y_train.append(data_cell)
    
    return np.float16(np.asarray(x_train)), np.float16(np.asarray(y_train))

  def __single_sample(self,
                      addr,
                      pos):
   
 
   if pos = "output":
 
    Ux, Uy, p, nuTilda = self.__get_domain(addr)
 
    Ux = self.__map_domain(Ux)
    Uy = self.__map_domain(Uy)
    p  = self.__map_domain(p)
 
    Ux = Ux.reshape([Ux.shape[0], Ux.shape[1], 1])
    Uy = Uy.reshape([Uy.shape[0], Uy.shape[1], 1])
    p =  p.reshape( [p.shape[0],  p.shape[1],  1])
 
    if (self.grid == "channel_flow"):
 
      height = self.height
      Ux_avg = Ux[int(height/2),0,0]
      Uy_avg = Uy[int(height/2),0,0]
      uavg = Ux_avg
      nuTildaAvg = 1e-4
  
    elif (self.grid == "ellipse"):
  
      uavg = 0.6
      nuTildaAvg = 1e-3
  
    Ux /= uavg
    Uy /= uavg
    p /= uavg*uavg
  
    data    = np.concatenate( (Ux, Uy) , axis=2)  
    data    = np.concatenate( (data, p), axis=2)
  
    if (self.is_turb):
     nuTilda = self.__map_domain(nuTilda)
     nuTilda = nuTilda.reshape([nuTilda.shape[0], nuTilda.shape[1], 1])
     nuTilda /= nuTildaAvg
     data   = np.concatenate( ( data, nuTilda), axis=2) 
  
 
   elif pos = "input":
   
    Ux, Uy, _, _ = self.__get_domain(addr)
 
    Ux = self.__map_domain(Ux)
    Uy = self.__map_domain(Uy)
 
    Ux = Ux.reshape([Ux.shape[0], Ux.shape[1], 1])
    Uy = Uy.reshape([Uy.shape[0], Uy.shape[1], 1])
  
    x,z = self.xyz[:,0], self.xyz[:,2]
    x,z = self.__map_domain(x), self.__map_domain(z)
    x = x.reshape([x.shape[0], x.shape[1],1])
    z = z.reshape([z.shape[0], z.shape[1],1])

    x /= 500
    z /= 500
  
    u = Ux[-1,-1,0]
    v = Uy[-1,-1,0]
    U = np.sqrt(u*u+v*v)
    alpha = np.arctan(v/u)
    alpha = alpha/np.pi*180
    Re = np.full_like(x,fill_value=U)
    alpha = np.full_like(x,fill_value=alpha)
  
    data    = np.concatenate( (x, z) , axis=2)  
    data    = np.concatenate( (data, Re) , axis=2)  
    data    = np.concatenate( (data, alpha) , axis=2)  
  
   return np.float16(data)
 

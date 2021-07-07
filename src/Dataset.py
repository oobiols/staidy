import os
import h5py
import glob
import openfoamparser as Ofpp
import numpy as np
import tensorflow as tf

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
              size= [64,256],
              grid = "ellipse",
              is_turb=1,
              add_coordinates = 0):

  self.height       = size[0]
  self.width        = size[1]
  self.channels     = 4
  self.is_turb      = is_turb
  self.add_coordinates = add_coordinates
  if self.add_coordinates: self.channels = 6
  self.directory    = './h5_datasets/'
  self.dataset_name = 'coarse_grid'
  self.dataset_type = 'train'
  self.grid         = grid
  self.path         = self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'
  self.shape        = [None,self.height,self.width,self.channels]
  self.shape_input = self.shape
  self.shape_output = self.shape
  
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

 def extract_2d_patches(self,images,patch_size):
  
      nRowsImage = images.shape[1]
      nColumnsImage = images.shape[2]
      nPixelsImage = nRowsImage*nColumnsImage

      nRowsPatch = patch_size[0]
      nColumnsPatch = patch_size[1]
      nPixelsPatch = nRowsPatch*nColumnsPatch

      nPatchImage = (nPixelsImage // nPixelsPatch)

      batch_size = tf.shape(images)[0]
      channels = tf.shape(images)[3]

      patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size[0], patch_size[1], 1],
            strides=[1, patch_size[0], patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

      patch_dims = patches.shape[-1]
      patches = tf.reshape(patches, [batch_size, -1, patch_dims])
      patches = tf.reshape(patches, [patches.shape[0],patches.shape[1],patch_size[0], patch_size[1],channels])
      patches = patches.numpy()

      return patches



 def load_data(self,cases,patches=0,patch_size=[32,128]):

  X=np.empty([0,self.height,self.width,self.channels],dtype=np.float16)
  Y=np.empty([0,self.height,self.width,self.channels],dtype=np.float16)

  for case in cases:

   self.file_path = self.directory+self.dataset_type+'/'+str(self.height)+'x'+ str(self.width)+'/'+case+'.h5'
   h5f = h5py.File(self.file_path,"r")

   x = h5f.get('x')
   x = np.asarray(x)
   y = h5f.get('y')
   y = np.asarray(y)

   x[:,:,:,3] = 1e-3*x[:,:,:,3]/1e-4
   y[:,:,:,3] = 1e-3*y[:,:,:,3]/1e-4

   if self.add_coordinates==False:
    x = x[:,:,:,0:4]
    y = y[:,:,:,0:4]

   X = np.append(X,x,axis=0)
   Y = np.append(Y,y,axis=0)

  if patches: 
      X = self.extract_2d_patches(X,patch_size)
      Y = self.extract_2d_patches(Y,patch_size)

  return X,Y


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
  h5f.create_dataset('x', data = x,compression="lzf",chunks=True, maxshape = self.shape_input)
  h5f.create_dataset('y', data = y,compression="lzf", chunks=True, maxshape = self.shape_output)
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
     self.xyz = self.get_coordinates(self.dataset_type,case)
    #Reads the path addresses where the OpenFOAM data is stored
    train_x_addrs, train_y_addr = self.read_addrs(self.dataset_type, case)
    #Reads the primary variable values at each iteration of the OpenFOAM data and t 
    train_x, train_y = self.__case_data(train_x_addrs, train_y_addr)
  
    print("x size ",train_x.shape)
    print("y size ",train_y.shape)

    if count==0:

     h5f = h5py.File(self.file_path,"w")
     h5f.create_dataset('x', data = train_x,compression="gzip",compression_opts=5,chunks=True, maxshape = self.shape_input)
     h5f.create_dataset('y', data = train_y,compression="gzip",compression_opts=5, chunks=True, maxshape = self.shape_output)
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


 def get_coordinates(self,
			data,
			case):

  
  coord_path   = "./" + data+"_data" + "/" + case + "/"
  xyz = np.loadtxt(coord_path+"xyz.txt")

  return xyz


 def read_addrs(self,
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

  Ux = self.map_domain(Ux)
  Uy = self.map_domain(Uy)
  p  = self.map_domain(p)

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
   nuTilda = self.map_domain(nuTilda)
   nuTilda = nuTilda.reshape([nuTilda.shape[0], nuTilda.shape[1], 1])
   nuTilda /= nuTildaAvg
   data   = np.concatenate( ( data, nuTilda), axis=2) 

  if (self.add_coordinates):
    x,z = self.xyz[:,0], self.xyz[:,2]
    x,z = self.map_domain(x), self.map_domain(z)
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


 def convert_to_foam(self,arr):

#  p0 = arr[:,0,:,:,:]
#  p1 = arr[:,1,:,:,:]
#  p2 = arr[:,2,:,:,:]
#  p3 = arr[:,3,:,:,:]
#  
#  bottom = np.append(p0,p2,axis=1)
#  top = np.append(p1,p3,axis=1)
#  arr = np.append(bottom,top,axis=2)

  Ux = arr[:,:,:,0]
  Uz = arr[:,:,:,0]
  p = arr[:,:,:,0]
  nuTilda = arr[:,:,:,0]

  Ux = self.unmap_domain(Ux)
  Uz = self.unmap_domain(Uz)
  p = self.unmap_domain(p)
  nuTilda = self.unmap_domain(nuTilda)

  Uy = np.full_like(Ux,fill_value=0)
  self.vector_to_foam(Ux,Uy,Uz,"U")
  self.pressure_to_foam(p,"p")
  self.nuTilda_to_foam(nuTilda,"nuTilda")


 def vector_to_foam(self,X,Y,Z,variable_name="U"):

  #Uavg = 0.6
  Uavg=1.0
  n_samples = len(X)
  for n in range(n_samples):

   directory_path='./predicted_fields/'+str(n)+'/'
   if not os.path.exists(directory_path):
      os.makedirs(directory_path)

   x = X[n] 
   y = Y[n]
   z = Z[n]

   f = open(directory_path+variable_name, "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volVectorField;\n	location	0;\n	object	U;\n}\n")
   f.write("dimensions [0 1 -1 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<vector>\n" + str(x.shape[0]) + "\n(\n")
   for j in range(0,x.shape[0]): 
    f.write("("+repr(x[j]*Uavg)+" 0 "+repr(z[j]*Uavg)+")\n")
   f.write(");\n")
   f.write("boundaryField\n{\n")
   f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform (0.6 0 0);\n}")
   f.write("\nbottom\n{\n\ttype\t noSlip;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
   f.write("}\n")
  

 def pressure_to_foam(self,X, variable_name='p'):

  Uavg = 0.6
  #Uavg = 1.0
  n_samples = len(X)
  for n in range(n_samples):

   directory_path='./predicted_fields/'+str(n)+'/'
   if not os.path.exists(directory_path):
      os.makedirs(directory_path)

   x = X[n] 

   f = open(directory_path+variable_name, "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	p;\n}\n")
   f.write("dimensions [0 2 -2 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<scalar>\n" + str(x.shape[0]) + "\n(\n")
   for j in range(0,x.shape[0]): 
    f.write(repr(x[j]*(Uavg*Uavg)) +"\n")
   
   f.write(");\n")
   f.write("boundaryField\n{\n")
   f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 0;\n}")
   f.write("\nbottom\n{\n\ttype\t zeroGradient;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

 def nuTilda_to_foam(self,X, variable_name='nuTilda'):

  Uavg = 1e-4
  #Uavg = 1.0
  n_samples = len(X)
  for n in range(n_samples):

   directory_path='./predicted_fields/'+str(n)+'/'
   if not os.path.exists(directory_path):
      os.makedirs(directory_path)

   x = X[n] 

   f = open(directory_path+variable_name, "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	nuTilda;\n}\n")
   f.write("dimensions [0 2 -1 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<scalar>\n" + str(x.shape[0]) + "\n(\n")
   for j in range(0,x.shape[0]): 
    f.write(repr(x[j]*(Uavg)) +"\n")
   
   f.write(");\n")
   f.write("boundaryField\n{\n")
   f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 3e-6;\n}")
   f.write("\nbottom\n{\n\ttype\t fixedValue;\nvalue\t uniform 0;}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
   f.write("}\n")

 def unmap_domain(self,var):

  n_samples = var.shape[0]
  height = var.shape[1]
  width = var.shape[2]
  w = int(width/4)
  variable_list = []

  for n in range(n_samples):
   
    arr = var[n,:,:] 
  
    b_1 = np.empty([0, width], float)
    b_2 = np.empty([0, width], float)
    b_3 = np.empty([0, width], float)
    b_4 = np.empty([0, width], float)

    for i in range (0,height,4):

      line1 = arr[i:i+4,3*w:4*w]
      line1 = line1.reshape([1,width])
      b_1 = np.append(b_1,line1,axis=0)

      line2 = arr[i:i+4,2*w:3*w]
      line2 = line2.reshape([1,width])
      b_2 = np.append(b_2,line2,axis=0)
      
      line4 = arr[i:i+4,w:2*w]
      line4 = np.flip(line4,axis=1)
      line4 = line4.reshape([1,width])
      b_4 = np.append(b_4,line4,axis=0)
 
      line3 = arr[i:i+4,0:w]
      line3 = np.flip(line3,axis=1)
      line3 = line3.reshape([1,width])
      b_3 = np.append(b_3,line3,axis=0)
    
    b_1 = b_1.reshape([height * w])
    b_2 = b_2.reshape([height * w])
    b_3 = b_3.reshape([height * w])
    b_4 = b_4.reshape([height * w])

    arr = np.append(b_1,b_2)
    arr = np.append(arr,b_3)
    arr = np.append(arr,b_4)

    variable_list.append(arr)

  return variable_list

 def map_domain(self, arr):

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
  def __init__(self,**kwargs):
    super(DatasetNoWarmup, self).__init__(**kwargs)
    self.add_coordinates=True
    self.shape_input=[None,self.height,self.width,self.channels]
    self.shape_output=[None,self.height,self.width,self.channels]

  def __case_data(self,
                 x_addrs, 
                 y_addr):

    x_train = []
    y_train = []
  
    x_addr = x_addrs[0]
    y_addr = y_addr[0]
     
    data_cell  = self.__single_sample(x_addr[0],"input")
    
    x_train.append(data_cell)
  
    data_cell = self.__single_sample(y_addr[0],"output")
  
    y_train.append(data_cell)
    
    return np.float16(np.asarray(x_train)), np.float16(np.asarray(y_train))

  def __single_sample(self,
                      addr,
                      pos):
   
   if pos == "output":
 
    Ux, Uy, p, nuTilda = self.__get_domain(addr,pos)
 
    Ux = self.map_domain(Ux)
    Uy = self.map_domain(Uy)
    p  = self.map_domain(p)
 
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
     nuTilda = self.map_domain(nuTilda)
     nuTilda = nuTilda.reshape([nuTilda.shape[0], nuTilda.shape[1], 1])
     nuTilda /= nuTildaAvg
     data   = np.concatenate( ( data, nuTilda), axis=2) 
  
    x,z = self.xyz[:,0], self.xyz[:,2]
    x,z = self.map_domain(x), self.map_domain(z)
    x = x.reshape([x.shape[0], x.shape[1],1])
    z = z.reshape([z.shape[0], z.shape[1],1])

    data    = np.concatenate( (data, x) , axis=2)  
    data    = np.concatenate( (data, z) , axis=2)  
 
   elif pos == "input":
   
    u, v, _, _ = self.__get_domain(addr,pos)
 
    x,z = self.xyz[:,0], self.xyz[:,2]
    x,z = self.map_domain(x), self.map_domain(z)
    x = x.reshape([x.shape[0], x.shape[1],1])
    z = z.reshape([z.shape[0], z.shape[1],1])

    x /= 500
    z /= 500
  
    U = np.sqrt(u*u+v*v)
    alpha = np.arctan(v/u)
    alpha = alpha/np.pi*180
    Re = np.full_like(x,fill_value=U)
    alpha = np.full_like(x,fill_value=alpha)
  
    data    = np.concatenate( (x, z) , axis=2)  
    data    = np.concatenate( (data, Re) , axis=2)  
    data    = np.concatenate( (data, alpha) , axis=2)  
  
   return np.float16(data)
 
  def __get_domain(self,
                    addr, 
		pos):


   if pos =="output":

    U = np.float16(Ofpp.parse_internal_field(addr+"/U"))
    p = np.float16(Ofpp.parse_internal_field(addr+"/p"))
 
    if (self.is_turb):
     nuTilda     = np.float16(Ofpp.parse_internal_field(addr+"/nuTilda"))
    else:
     nuTilda = 0
 
    Ux          = U[:,0]
    Uy          = U[:,2]

   elif pos=="input":

    p = None
    nuTilda = None
    U = Ofpp.parse_boundary_field(addr+'/U')
    U=U[b'top']
    U=U[b'freestreamValue']
    Ux = U[0]
    Uy = U[2]

   return Ux, Uy, p, nuTilda

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

     if (np.mod(case_number,10)!=0): 
      case = "case_"+str(case_number)
      print("case number is ", case)
  
      if self.add_coordinates:
       self.xyz = self.get_coordinates(self.dataset_type,case)
      #Reads the path addresses where the OpenFOAM data is stored
      train_x_addrs, train_y_addr = self.read_addrs(self.dataset_type, case)
      #Reads the primary variable values at each iteration of the OpenFOAM data and t 
      train_x, train_y = self.__case_data(train_x_addrs, train_y_addr)
    
      print("x size ",train_x.shape)
      print("y size ",train_y.shape)
  
      if count==0:
  
       h5f = h5py.File(self.file_path,"w")
       h5f.create_dataset('x', data = train_x,compression="gzip",compression_opts=5,chunks=True, maxshape = self.shape_input)
       h5f.create_dataset('y', data = train_y,compression="gzip",compression_opts=5, chunks=True, maxshape = self.shape_output)
       h5f.close() 
  
      else:
  
       with h5py.File(self.file_path, 'a') as hf:
  
        hf["x"].resize((hf["x"].shape[0] + train_x.shape[0]), axis = 0)
        hf["x"][-train_x.shape[0]:] = train_x
  
        hf["y"].resize((hf["y"].shape[0] + train_y.shape[0]), axis = 0)
        hf["y"][-train_y.shape[0]:] = train_y
         
        hf.close()
   
     else:
      print("No case ",case_number)
  
     case_number=case_number+1
     count=count+1  
  
    return 

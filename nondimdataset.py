import sys
sys.path.append('./src')
from Dataset import *
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='dataset command line')

parser.add_argument('-t','--type', type=str, default="train", help="Options are: 'train', 'validation' or 'test' ")
parser.add_argument('-n','--name', type=str, default="coarse_grid", help='Desired name of the dataset')
parser.add_argument('-he','--height', type=int, default=32,help="Height of the computational domain. If domain is HxW, this is H")
parser.add_argument('-w','--width',type=int, default="128", help= "Width of the computational domain. If mesh is HxW, this is W")
args = parser.parse_args()

h=args.height
w=args.width
c=3
size=[h,w,c]
ds = DatasetNoWarmup(size=size, 
	     add_coordinates = 1)

ds.set_type(args.type)
cases=[args.name]

x  , _  = ds.load_data(cases)

n = x.shape[0]

X=np.empty([0,args.height,args.width,8],dtype=np.float16)

for i in range(n):
 
 ur  = x[i,args.height//2,0,0]
 pr = ur*ur
 nur = 2.5e-3
  
 u = x[i:i+1,:,:,0] / ur
 v = x[i:i+1,:,:,1] / ur
 p = x[i:i+1,:,:,2] / pr
 nut = x[i:i+1,:,:,3] / nur
 
 xc = x[i:i+1,:,:,4] 
 zc = x[i:i+1,:,:,5] 

 R =  nur/ur
 nu = 1e-4/nur

 Re = np.full_like(xc,fill_value= R)
 nu = np.full_like(Re,fill_value=nu)

 data = np.stack([u,v,p,nut,xc,zc,Re,nu],axis=-1)
 X = np.append(X,data,axis=0)

print(X.shape)
np.save('./datasets/'+args.name+'_lr_turb_nondim.npy',X)

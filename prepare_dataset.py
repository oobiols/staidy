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
parser.add_argument('-nd','--nondimensional',type=int, default=1, help= "Width of the computational domain. If mesh is HxW, this is W")
args = parser.parse_args()

h=args.height
w=args.width
c=3
size=[h,w,c]
ds = DatasetNoWarmup(size=size, 
	     add_coordinates = 1)

ds.set_type(args.type)
cases=[args.name]

x  , y = ds.load_data(cases)

print(x.shape)

if (args.nondimensional):

 ur = 5
 pr = ur*ur
 nur = 1e-2
 lr = 1
 
 x[:,:,:,0] /= ur
 x[:,:,:,1] /= ur
 x[:,:,:,2] /= pr
 x[:,:,:,3] /= nur
 x[:,:,:,4] /= lr
 x[:,:,:,5] /= lr


np.save('./datasets/'+args.name+'_lr_turb.npy',x)

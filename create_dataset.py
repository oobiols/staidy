import sys
sys.path.insert(0, './src')

import argparse

from Dataset import *

parser = argparse.ArgumentParser(description='dataset command line')

parser.add_argument('-t','--type', type=str, default="train", help="Options are: 'train', 'validation' or 'test' ")
parser.add_argument('-n','--name', type=str, default="coarse_grid", help='Desired name of the dataset')
parser.add_argument('-f','--turb', type=int, default=1,help="0 for laminar flow, 1 for turbulent flow (RANS - include eddy viscosity)")
parser.add_argument('-he','--height', type=int, default=32,help="Height of the computational domain. If domain is HxW, this is H")
parser.add_argument('-w','--width',type=int, default="128", help= "Width of the computational domain. If mesh is HxW, this is W")
parser.add_argument('-g','--grid',type=str, default="ellipse", help= "grid type for flow variable mapping. Current possible values accept either channel_flow or ellipse (external aerodynamcis cases)")
parser.add_argument('-lc','--lastcase',type=int, default=1, help= "last case number in this dataset")
parser.add_argument('-c','--coordinates',type=int, default=1, help= "last case number in this dataset")
args = parser.parse_args()

if (args.turb == 1 and args.coordinates == 1):
 channels = 6
elif (args.turb==1 and args.coordinates == 0):
 channels = 4
elif (args.turb==0 and args.coordiantes == 1):
 channels = 5
elif (args.turb==0 and args.coordiantes == 0):
 channels = 3



size = [args.height,args.width,channels]

ds = DatasetNoWarmup(size=size, 
             grid = args.grid,
             is_turb=args.turb,
	     add_coordinates = args.coordinates)

ds.set_name(args.name)
ds.set_type(args.type)
ds.create_dataset(last_case=args.lastcase)


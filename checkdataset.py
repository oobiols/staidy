import sys
sys.path.append('./src')
from Dataset import Dataset
import numpy as np

h = 32
w = 128

ds = Dataset(size=[h,w,6], 
	     add_coordinates = 1)

ds.set_type("train")
cases=["channelflow"]

X , Y = ds.load_data(cases,patches=0)

np.save('channelflow_LR',X)


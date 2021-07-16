import sys
sys.path.append('./src')
from Dataset import Dataset
import numpy as np

h=64
w=256
c=6
size=[h,w,c]
ds = Dataset(size=size, 
	     add_coordinates = 0)

ds.set_type("train")
cases=["ellipse01","NACA0012"]
X  , Y = ds.load_data(cases)
print(X.shape)


np.save('X_val.npy',X)
np.save('Y_val.npy',Y)


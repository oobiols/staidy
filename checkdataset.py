import sys
sys.path.append('./src')
from Dataset import Dataset
import numpy as np

ds = Dataset(size=[64,256,6], 
	     add_coordinates = 1)

ds.set_type("validation")
cases=["ellipse03"]

_ , Y_train = ds.load_data(cases,patches=0)

print(Y_train.shape)

Y=np.empty([0,Y_train.shape[1],Y_train.shape[2],Y_train.shape[3]],dtype=np.float16)
a = np.arange(1,Y_train.shape[0],1)

Y_old = Y_train[0,:,:,:].reshape([1,Y_train.shape[1],Y_train.shape[2],Y_train.shape[3]])
Y = np.append(Y,Y_old,axis=0)
print("Finding...")
for i in a:
 print(i)
 Y_new = Y_train[i,:,:,:].reshape([1,Y_train.shape[1],Y_train.shape[2],Y_train.shape[3]])
 equal = np.array_equal(Y_new,Y_old)
 if equal == False:
   Y = np.append(Y,Y_new,axis=0)


 Y_old = Y_new

print(Y.shape)
np.save('./superresolution_validation',Y)

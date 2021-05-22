import numpy as np

def MinMaxScaler(X):

 for i in range(X.shape[3]):

   maxvalue = np.max(X[:,:,:,i])
   minvalue = np.min(X[:,:,:,i])
   X[:,:,:,i] = np.divide(np.subtract(X[:,:,:,i],minvalue),maxvalue-minvalue)

   
 return X

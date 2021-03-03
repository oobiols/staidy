import h5py
import numpy as np

from sklearn.utils import shuffle

def loadTrainDataset(path):

  h5f = h5py.File(path,'r')

  train_x = h5f.get('x_train_dataset')
  train_x = np.float32(np.array(train_x))
  train_y = h5f.get('y_train_dataset')
  train_y = np.float32(np.array(train_y))

  return (train_x, train_y)

def loadTestDataset(path):

  h5f = h5py.File(path,'r')

  test_x = h5f.get('x_test_dataset')
  test_x = np.array(test_x)
  test_y = h5f.get('y_test_dataset')
  test_y = np.array(test_y)

  return (test_x, test_y)

def loadPredictDataset(path,i):
	
  i=str(i)
  h5f = h5py.File(path,'r')
  predict_x = h5f.get('x_predict_dataset_'+i)
  predict_x = np.array(predict_x)

  predict_y = h5f.get('y_predict_dataset_'+i)
  predict_y = np.array(predict_y)

  return (predict_x, predict_y)

def loadUniquePredictDataset(path):
	
  h5f = h5py.File(path,'r')
  predict_x = h5f.get('x_predict_dataset')
  predict_x = np.array(predict_x)

  predict_y = h5f.get('y_predict_dataset')
  predict_y = np.array(predict_y)

  return (predict_x, predict_y)

def loadrePredictDataset():
	

  h5f = h5py.File('dataset.h5','r')
  predict_x = h5f.get('repredict_x')
  predict_x = np.array(predict_x)

  predict_y = h5f.get('repredict_y')
  predict_y = np.array(predict_y)

  return (predict_x, predict_y)

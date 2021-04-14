import numpy as np
import tensorflow as tf
import h5py

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path='../datasets/train/ellipse/2048x2048/train.h5', batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.h5f = h5py.File(path,'r')
        self.L =  self.h5f.get("x").shape[0]

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.L / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        first = indexes[0]
        last = indexes[-1]+1
        # Generate data
        X, Y = self.__data_generation(first,last)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.L)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,first,last):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.float16(self.h5f["x"][first:last])
        Y = np.float16(self.h5f["y"][first:last])

        return X, Y


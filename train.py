import sys
sys.path.insert(0, './func/')
sys.path.insert(0, './datasets/')

from Dataset import *
from models import *

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

mirrored_strategy = tf.distribute.MirroredStrategy()

parser = argparse.ArgumentParser(description='dataset command line')


parser.add_argument('-e','--epochs', type=int, default=50, help="Number of epochs to train for")
parser.add_argument('-b','--batchsize', type=int, default=64, help="Batch size")
parser.add_argument('-he','--height', type=int, default=32,help="Height of the computational domain. If domain is HxW, this is H")
parser.add_argument('-w','--width',type=int, default="128", help= "Width of the computational domain. If mesh is HxW, this is W")

args = parser.parse_args()

#training image size (resolution of the coarse-grid data)
height=args.height
width=args.width
channels=4
input_shape = (height,width, channels)
size = [ height, width, channels]


#load training dataset
ds_tr = Dataset(size)
ds.set_directory("./h5_datasets/")
ds.set_type("train")
X_train,Y_train = ds.load_dataset()


#load validation dataset 
ds_val = Dataset(size)
ds_val.set_directory('./h5_datasets/')
ds_val.set_type("validation")
X_val , Y_val = ds_val.load_dataset()


#distribute to the number of available gpus
with mirrored_strategy.scope():

 cnn = NeuralNetwork(input_shape)
 cnn.setarchitecture_deep(sizefilter=(5,5),stride1=(1,1), stride2=(1,1),filter1=32,filter2=256,alpha=0.1,lamreg=0)
 cnn.create_model()
 cnn.compile_model()


name="Adam-1e-3-deep-RLROP"
checkpoint_filepath = './weights/'+name+'-checkpoint-{epoch:02d}'
 

callbacks=[ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=10, verbose=1,
    mode='auto', min_delta=0.01, cooldown=0, min_lr=0, **kwargs
)]

callbacks=[]

cnn.fit_model(X_train, 
              Y_train,
              X_val,
              Y_val,
              batch_size=args.batchsize,
              epochs=args.epochs,
              shuffle=True, 
            callbacks=callbacks)


  

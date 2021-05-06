import sys
sys.path.append('./src/')
import plot
import tensorflow as tf
import h5py
from Dataset import Dataset
import tensorflow as tf
from tensorflow import keras
import numpy as np
from NS_transformer import *
from losses import *

tf.keras.backend.set_floatx('float32')
image_size = [64,256,6]
patch_size=[32,128]
nRowsPatch = patch_size[0]
nColumnsPatch = patch_size[1]

nPixelsImage = image_size[0]*image_size[1]
nPixelsPatch = nRowsPatch * nColumnsPatch

nPatchesImage = nPixelsImage // nPixelsPatch

channelsOutput = image_size[2]
channelsInput  = image_size[2]

projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
mlp_head_units = [2048, 1024]
transformer_layers = 1

ds = Dataset(size=image_size, 
	     add_coordinates = 1)

ds.set_type("train")
ds.set_name("ellipse045")
X, Y = ds.load_dataset()
#X , Y = extract_2d_patches(X,patch_size,masking), extract_2d_patches(Y,patch_size,0)

print(X.shape)
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():

 inputs = keras.layers.Input(shape=(64,256,6))
 patches = Patches(patch_size)(inputs)
 encoded_patches = PatchEncoder(4,projection_dim)(patches)
 
 for _ in range(transformer_layers):
 
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    attention_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1,x1)
    x2 = keras.layers.Add()([attention_output, encoded_patches])
    x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    encoded_patches = keras.layers.Add()([x3, x2])
 
 representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
 representation = keras.layers.Flatten()(representation)
 representation = keras.layers.Dropout(0.5)(representation)
 
 features = keras.layers.Dense(nPatchesImage*nPixelsPatch*4)(representation)
 features = keras.layers.LeakyReLU(alpha=0.2)(features)
 reshape = keras.layers.Reshape((int(nPatchesImage),int(nRowsPatch),int(nColumnsPatch),int(4)))(features)
 conv2d_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1),activation="linear")(reshape)

 model = keras.Model(inputs=inputs, outputs=conv2d_1)
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=mse_total)


Y = Y[:,:,:,0:4]
Y = Patches(patch_size)(Y)
Y = tf.reshape(Y,[Y.shape[0],Y.shape[1],nRowsPatch,nColumnsPatch,4])
print(Y.shape)
model.fit(x=X,y=Y,epochs=100,batch_size=64,verbose=1,validation_data=(X,Y),shuffle=True)


import tensorflow as tf
from tensorflow import keras
import numpy as np

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def corrupt_patches(patches,nPatchesImage,masking):

    batch_size = patches.shape[0]

    for i in range(batch_size):
     pn = np.random.randint(0,high=nPatchesImage,size=1,dtype=int)[0]
     patches[i,pn,:,:,:] = np.random.randn()

    return patches

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim,activation=tf.nn.leaky_relu,trainable=True,name ="dense_projection")
        self.position_embedding = keras.layers.Embedding(
                                                            input_dim=num_patches, 
                                                            output_dim=projection_dim,
                                                            trainable=True,
                                                            name = "pos_embedding")

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class VisionTransformerLayers(keras.layers.Layer):
  def __init__(self, 
               image_size=[64,256,6],
               patch_size = [32,128],
               projection_dim_encoder=768,
               projection_dim_attention=64,
               num_heads = 4,
               transformer_layers=12,
               **kwargs):

    super(VisionTransformerLayers, self).__init__(**kwargs)

    self.image_size = image_size
    self.patch_size = patch_size
    
    self.channelsInput = image_size[2]
    self.channelsOutput = 4

    self.nRowsImage = image_size[0]
    self.nColumnsImage = image_size[1]
    self.nPixelsImage = self.nRowsImage * self.nColumnsImage

    self.nRowsPatch = patch_size[0]
    self.nColumnsPatch = patch_size[1]
    self.nPixelsPatch = self.nRowsPatch * self.nColumnsPatch

    self.nPatchesImage = ( ( self.nPixelsImage  ) // (self.nPixelsPatch) )

    self.projection_dim_encoder = projection_dim_encoder
    self.projection_dim_attention = projection_dim_attention
    self.num_heads = num_heads
    self.transformer_layers = transformer_layers
    self.transformer_units = [
    self.projection_dim_encoder * 2,
    self.projection_dim_encoder,
                         ]
    self.Norm0 = []
    self.Norm1 = []
    self.Attention = []
    self.Dense0 = []
    self.Dense1 = []
    self.Dropout0 = []
    self.Dropout1 = []
    self.Add0 = []
    self.Add1 = []

    self.Patches = keras.layers.Reshape( (int(self.nPatchesImage), int(self.nPixelsPatch*self.channelsInput) ),name="reshape1" ) #[BS,NP,PATCH_DIM] PATCH_DIM = Npixe
    self.EncodedPatches = PatchEncoder(self.nPatchesImage,self.projection_dim_encoder) #[BS,NP,PROJ_DIM]

    
    #transformer encoder
    for i in range(transformer_layers):
        
        self.Norm0.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=True,name="Trans/norm0"))
        self.Attention.append(keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim_attention, dropout=0.1,trainable=True,name="Trans/Attention"))
        self.Add0.append(keras.layers.Add(name="Trans/Add0"))
        self.Norm1.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=True,name="Trans/Norm1"))
        self.Dense0.append(keras.layers.Dense(4*self.projection_dim_encoder,activation= tf.nn.leaky_relu,trainable=True,name="Trans/Dense0"))
        self.Dropout0.append(keras.layers.Dropout(rate=0.1))
        self.Dense1.append(keras.layers.Dense(self.projection_dim_encoder,activation= tf.nn.leaky_relu,trainable=True,name="Trans/Dense1"))
        self.Dropout1.append(keras.layers.Dropout(rate=0.1))
        self.Add1.append(keras.layers.Add(name="Trans/Add1"))

    #map to steady state
    self.Flatten = keras.layers.Flatten(name="Task/Flatten")
    self.MapDense = keras.layers.Dense(256,activation=tf.nn.leaky_relu,name="Map/Dense",trainable=True )
    self.MapReshape0 = keras.layers.Reshape( (4,16,self.channelsOutput),name="Task/Reshape" )
    self.MapDeconv = keras.layers.Conv2DTranspose(filters=4*self.channelsOutput,kernel_size=(3,3),padding="same",strides=(4,4),activation=tf.nn.leaky_relu,name="Task/Conv",trainable=True)
    self.MapDeconv2 = keras.layers.Conv2DTranspose(filters=self.channelsOutput,kernel_size=(3,3),padding="same",strides=(8,8),activation='linear',name="Task/Conv2",trainable=True)
    #self.MapReshape1 = keras.layers.Reshape( (self.nPatchesImage, self.nRowsPatch, self.nColumnsPatch, self.channelsOutput),name="Map/Rehsape_Out" )


  def call(self, inputs):

    patches = self.Patches(inputs)
    encoded_patches = self.EncodedPatches(patches)

    for i in range(self.transformer_layers):

     x1 = self.Norm0[i](encoded_patches)
     attention = self.Attention[i](x1,x1)
     x2 = self.Add0[i]([attention,encoded_patches]) 
     x3 = self.Norm1[i](x2)
     x3 = self.Dense0[i](x3)
     x3 = self.Dropout0[i](x3)
     x3 = self.Dense1[i](x3)
     x3 = self.Dropout1[i](x3)
     encoded_patches = self.Add1[i]([x3,x2])

    #map to steady-state patches 
    x = self.Flatten(encoded_patches)
    x = self.MapDense(x)
    x = self.MapReshape0(x)
    x = self.MapDeconv(x)
    x = self.MapDeconv2(x)
  #  x = self.MapReshape1(x)
    
    return x 

import tensorflow as tf
from tensorflow import keras
import numpy as np

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def extract_2d_patches(images,patch_size=[32,128],masking=0):

      nRowsImage = images.shape[1]
      nColumnsImage = images.shape[2]
      nPixelsImage = nRowsImage*nColumnsImage

      nRowsPatch = patch_size[0]
      nColumnsPatch = patch_size[1]
      nPixelsPatch = nRowsPatch*nColumnsPatch

      nPatchImage = (nPixelsImage // nPixelsPatch)

      batch_size = tf.shape(images)[0]
      channels = tf.shape(images)[3]

      patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size[0], patch_size[1], 1],
            strides=[1, patch_size[0], patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

      patch_dims = patches.shape[-1]
      patches = tf.reshape(patches, [batch_size, -1, patch_dims])
      patches = tf.reshape(patches, [patches.shape[0],patches.shape[1],patch_size[0], patch_size[1],channels])
      patches = patches.numpy()

      if(masking):
       patches = corrupt_patches(patches,nPatchImage,masking)

      return patches

def corrupt_patches(patches,nPatchesImage,masking):

    batch_size = patches.shape[0]

    for i in range(batch_size):
     pn = np.random.randint(0,high=nPatchesImage,size=1,dtype=int)[0]
     patches[i,pn,:,:,:] = np.random.randn()

    return patches

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim,activation=tf.nn.elu)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

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

    self.Patches = keras.layers.Reshape( (int(self.nPatchesImage), int(self.nPixelsPatch*self.channelsInput) ) )
    self.EncodedPatches = PatchEncoder(self.nPatchesImage,self.projection_dim_encoder)

    #transformer encoder
    for _ in range(transformer_layers):

        self.Norm0.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=True))
        self.Attention.append(keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim_attention, dropout=0.1,trainable=False))
        self.Add0.append(keras.layers.Add())
        self.Norm1.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=True))
        self.Dense0.append(keras.layers.Dense(4*self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False))
        self.Dropout0.append(keras.layers.Dropout(rate=0.1))
        self.Dense1.append(keras.layers.Dense(self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False))
        self.Dropout1.append(keras.layers.Dropout(rate=0.1))
        self.Add1.append(keras.layers.Add())

    
    #map to steady state
    self.Flatten = keras.layers.Flatten()
    self.MapDense = keras.layers.Dense(self.nPixelsPatch,activation=tf.nn.gelu)
    self.MapReshape0 = keras.layers.Reshape( (self.nRowsPatch//2,self.nColumnsPatch//2,self.channelsOutput) )
    self.MapDeconv = keras.layers.Conv2DTranspose(filters=self.channelsOutput,kernel_size=(5,5),padding="same",strides=(4,4),activation='linear')
    self.MapReshape1 = keras.layers.Reshape( (self.nPatchesImage, self.nRowsPatch, self.nColumnsPatch, self.channelsOutput) )

#    self.Flatten = keras.layers.Flatten()
#    self.Normalization_1 = keras.layers.LayerNormalization(epsilon=1e-6)
#    self.AttentionOutput = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)
#    self.Add_1 = keras.layers.Add()
#    self.Normalization_2 = keras.layers.LayerNormalization(epsilon=1e-6)
#    self.Dense_1 = keras.layers.Dense(self.projection_dim*2)
#    self.Activation_1 = keras.layers.LeakyReLU(alpha=0.2)
#    self.Dropout_1 = keras.layers.Dropout(rate=0.1)
#    self.Dense_2= keras.layers.Dense(self.projection_dim)
#    self.Activation_2 = keras.layers.LeakyReLU(alpha=0.2)
#    self.Dropout_2 = keras.layers.Dropout(rate=0.1)
#    self.Add_2 = keras.layers.Add()
#
#    self.Normalization_3 = keras.layers.LayerNormalization(epsilon=1e-6)
#    self.Flatten = keras.layers.Flatten()
#    self.Dense_3 = keras.layers.Dense(self.nPatchesImage*self.nPixelsPatch*self.channelsOutput)
#    self.Activation_3 = keras.layers.LeakyReLU(alpha=0.2)
#    self.Reshape = keras.layers.Reshape((int(self.nPatchesImage),int(self.nRowsPatch),int(self.nColumnsPatch),int(self.channelsOutput)))
#    self.Conv2D_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1),activation='linear')
#    self.Activation_4 = keras.layers.LeakyReLU(alpha=0.2)
   
   
  def call(self, inputs):

    patches = self.Patches(inputs)
    encoded_patches = self.EncodedPatches(patches)

    for i in range(self.transformer_layers):

     x1 = self.Norm0[i](encoded_patches)
     attention = self.Attention[i](x1,x1)
     x2 = self.Add0[i]([attention_output,encoded_patches]) 
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
    x = self.MapReshape1(x)
    
    return x 

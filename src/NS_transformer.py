import tensorflow as tf
from tensorflow import keras
import numpy as np

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def extract_2d_patches(images,patch_size,masking):

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

class TransformerLayers(keras.layers.Layer):
  def __init__(self, 
               image_size=[64,256,6],
               patch_size = [32,128],
               projection_dim=64,
               num_heads = 4,
               transformer_layers=1,
               masking=0,
               **kwargs):

    super(TransformerLayers, self).__init__(**kwargs)

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

    self.projection_dim = projection_dim
    self.num_heads = num_heads
    self.transformer_layers = transformer_layers
    self.transformer_units = [
    self.projection_dim * 2,
    self.projection_dim,
                         ]

    self.patches = keras.layers.Reshape( (int(self.nPatchesImage), int(self.nPixelsPatch*self.channelsInput) ) )
    #self.patches = Patches(self.patch_size)
    self.encoded_patches = PatchEncoder(self.nPatchesImage,self.projection_dim)
    
    #Transformer
    self.x1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.attention_output = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)
    self.x2 = keras.layers.Add()
    self.x3 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.dense_1 = keras.layers.Dense(self.projection_dim)
    self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.2)
    self.dropout = keras.layers.Dropout(rate=0.1)
    self.encoded_patches_transformer = keras.layers.Add()

    self.representation = keras.layers.LayerNormalization(epsilon=1e-6)
    self.flatten = keras.layers.Flatten()
    self.dense = keras.layers.Dense(self.nPatchesImage*self.nPixelsPatch*self.channelsOutput)
    self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.2)
    self.reshape = keras.layers.Reshape((int(self.nPatchesImage),int(self.nRowsPatch),int(self.nColumnsPatch),int(self.channelsOutput)))
    self.conv2d_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1))
    self.leaky_relu_3 = keras.layers.LeakyReLU(alpha=0.2)

  def call(self, inputs):

    patches = self.patches(inputs)
    encoded_patches = self.encoded_patches(patches)

    # transformer
    x1 = self.x1(encoded_patches)
    attention_output = self.attention_output(x1,x1)
    x2 = self.x2([attention_output,encoded_patches]) 
    x3 = self.x3(x2)
    x3 = self.dense_1(x3)
    x3 = self.leaky_relu_1(x3)
    x3 = self.dropout(x3)
    encoded_patches = self.encoded_patches_transformer([x3,x2])

    #map to steady-state patches 
    representation = self.representation(encoded_patches)
    x = self.flatten(representation)
    x = self.dense(x)
    x = self.leaky_relu_2(x)
    x = self.reshape(x)
    x = self.conv2d_1(x)
    x = self.leaky_relu_3(x)

    return x
    #patches = keras.layers.Reshape( (int(self.nPatchesImage), int(self.nPixelsPatch*self.channelsInput) ) )(inputs)

#    patches = Patches(self.patch_size)(inputs)
#    encoded_patches = PatchEncoder(self.nPatchesImage,self.projection_dim)(patches)
#
#    for _ in range(self.transformer_layers):
#
#       x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#       attention_output = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1,x1)
#       x2 = keras.layers.Add()([attention_output, encoded_patches])
#       x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
#       x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
#       encoded_patches = keras.layers.Add()([x3, x2])
#
#    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#    representation = keras.layers.Flatten()(representation)
#    representation = keras.layers.Dropout(0.5)(representation)
#    
#    features = keras.layers.Dense(self.nPatchesImage*self.nPixelsPatch*self.channelsOutput)(representation)
#    features = keras.layers.LeakyReLU(alpha=0.2)(features)
#    reshape = keras.layers.Reshape((int(self.nPatchesImage),int(self.nRowsPatch),int(self.nColumnsPatch),int(self.channelsOutput)))(features)
#    conv2d_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1))(reshape)
#    Leaky = keras.layers.LeakyReLU(alpha=0.2)(conv2d_1)
#    
    #return Leaky

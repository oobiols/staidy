import tensorflow as tf
from tensorflow import keras
import numpy as np

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
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

  def call(self, inputs):

    patches = keras.layers.Reshape( (int(self.nPatchesImage), int(self.nPixelsPatch*self.channelsInput) ) )(inputs)
    encoded_patches = PatchEncoder(self.nPatchesImage,self.projection_dim)(patches)

    for _ in range(self.transformer_layers):

       x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
       attention_output = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1,x1)
       x2 = keras.layers.Add()([attention_output, encoded_patches])
       x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
       x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
       encoded_patches = keras.layers.Add()([x3, x2])

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    
    features = keras.layers.Dense(self.nPatchesImage*self.nPixelsPatch*self.channelsOutput)(representation)
    reshape = keras.layers.Reshape((int(self.nPatchesImage),int(self.nRowsPatch),int(self.nColumnsPatch),int(self.channelsOutput)))(features)
    conv2d_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1), activation = 'linear')(reshape)

    return conv2d_1

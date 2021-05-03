import tensorflow as tf
from tensorflow import keras
import numpy as np

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def corrupt_patches(patches,num_patches):

    patches = patches.numpy()
    batch_size = patches.shape[0]

    for i in range(batch_size):
     pn = np.random.randint(0,high=num_patches,size=1,dtype=int)[0]
     patches[i,pn,:] = np.random.randn()

    patches = tf.convert_to_tensor(patches)

    return patches

class Patches(keras.layers.Layer):

    def __init__(self, patch_size, num_patches, masking=0):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.masking = masking

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
   
        if (self.masking):
            patches = corrupt_patches(patches,self.num_patches)
  
        return patches

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

class TransformerLayers(keras.Model):
  def __init__(self, 
                input_shape=[64,256,6],
                patch_size = [32,128],
                projection_dim=64,
                num_heads = 4,
                transformer_layers=1,
                masking=0,
                **kwargs):

    super(TransformerLayers, self).__init__(**kwargs)

    imagesize = input_shape[0]*input_shape[1]
    patchsize = tf.reduce_prod(patch_size)

    self.masking = masking
    self.input_shape = input_shape
    self.patch_size = patch_size
    self.num_patches = (imagesize // patchsize)
    self.projection_dim = projection_dim
    self.num_heads = num_heads
    self.transformer_layers = transformer_layers
    self.transformer_units = [
    self.projection_dim * 2,
    self.projection_dim,
                         ]

  def call(self, inputs):
 
    patches = Patches(self.patch_size,self.num_patches,self.masking)(inputs)
    encoded_patches = PatchEncoder(self.num_patches,self.projection_dim)(patches)

    for _ in self.transformer_layers:

       x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
       attention_output = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1)(x1,x1)
       x2 = layers.Add()([attention_output, encoded_patches])
       x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
       x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
       encoded_patches = layers.Add()([x3, x2])

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    
    features = keras.layers.Dense(self.num_patches*self.patch_size[0]*self.patch_size[1]*input_shape[2])(representation)
    reshape = keras.layers.Reshape((int(self.num_patches),int(self.patch_size[0]),int(self.patch_size[1]),int(4)))(features)
    conv2d_1 = keras.layers.Conv2D(filters=4,kernel_size = (5,5),padding = "same", strides = (1,1), activation = 'linear')(reshape)

    return conv2d_1

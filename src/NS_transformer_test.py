import tensorflow as tf
from tensorflow import keras
import numpy as np

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim,activation=tf.nn.gelu,trainable=False,name ="dense_projection")
        self.position_embedding = keras.layers.Embedding(
                                                            input_dim=num_patches, 
                                                            output_dim=projection_dim,
                                                            trainable=False,
                                                            name = "pos_embedding")

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class Projection(keras.layers.Layer):
    def __init__(self,projection_dim_encoder,patch_size):
     super(Projection,self).__init__()
     self.projection = keras.layers.Conv2D(filters=projection_dim_encoder,
                                           kernel_size=patch_size,
                                           strides=patch_size,
                                           padding="valid",
                                           name="Pre/Projection",
                                           trainable=False)
    def call(self,inputs):

      projection = self.projection(inputs)

      return projection

class PositionEmbedding(keras.layers.Layer):
    def __init__(self,sequence_length,projection_dim_encoder):
     super(PositionEmbedding,self).__init__()
     self.sequence_length = sequence_length
     self.position_embedding = keras.layers.Embedding(input_dim=sequence_length,
                                                      output_dim=projection_dim_encoder,
                                                      trainable=False,
                                                      name="Pre/PositionEmbedding")
    def call(self,inputs):
     positions = tf.range(start=0,limit=self.sequence_length,delta=1)
     embedding = self.position_embedding(positions)
     return inputs + embedding


#class masking


class NSTransformer(keras.Model):
  def __init__(self, 
               image_size=[64,256,6],
               filter_size=[16,16],
               patch_size = [32,128],
               sequence_length = 196,
               projection_dim_encoder=768,
               projection_dim_attention=64,
               num_heads = 12,
               transformer_layers=12,
               **kwargs):

    super(NSTransformer, self).__init__(**kwargs)

    self.image_size = image_size
    self.patch_size = patch_size
    self.filter_size = filter_size
    self.sequence_length = sequence_length

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

    self.preprocess()
    self.encoder()
    self.task()

  def preprocess(self):

    self.InitialReshape = keras.layers.Reshape((self.image_size[0],self.image_size[1],self.image_size[2]))
    self.InitialDeconv =keras.layers.Conv2DTranspose(filters=3,kernel_size=(49,193),strides=(1,1),padding="valid",name="Pre/Deconv")
    self.Projection = Projection(self.projection_dim_encoder,self.filter_size)
    self.Reshape = keras.layers.Reshape((self.sequence_length,self.projection_dim_encoder))
    self.PositionEmbedding = PositionEmbedding(self.sequence_length,self.projection_dim_encoder)


  def task(self):

    self.Flatten = keras.layers.Flatten(name="Task/Flatten")
    self.MapDense = keras.layers.Dense(16,activation=tf.nn.gelu,name="Task/Dense")
    self.MapReshape0 = keras.layers.Reshape( (2,2,self.channelsOutput),name="Task/Reshape" )
    self.MapDeconv = keras.layers.Conv2DTranspose(filters=self.channelsOutput,kernel_size=(5,5),padding="same",strides=(32,128),activation='linear',name="Task/Conv")
    self.MapReshape1 = keras.layers.Reshape( (self.nPatchesImage, self.nRowsPatch, self.nColumnsPatch, self.channelsOutput),name="Task/Reshape_Out" )

  def encoder(self):

      for i in range(self.transformer_layers):
 
        name = 'T/EB_'+str(i)+'/'

        self.Norm0.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=True,name=name+'Norm0'))
        self.Attention.append(keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim_attention, dropout=0.1,trainable=False,name = name+'Attention'))
        self.Add0.append(keras.layers.Add(name=name+"Add0"))
        self.Norm1.append(keras.layers.LayerNormalization(epsilon=1e-6,trainable=False,name=name+"Norm1"))
        self.Dense0.append(keras.layers.Dense(4*self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False,name=name+"Dense0"))
        self.Dropout0.append(keras.layers.Dropout(rate=0.1,name=name+'Dropout0'))
        self.Dense1.append(keras.layers.Dense(self.projection_dim_encoder,activation= tf.nn.gelu,trainable=False,name=name+"Dense1"))
        self.Dropout1.append(keras.layers.Dropout(rate=0.1,name=name+'Dropout1'))
        self.Add1.append(keras.layers.Add(name=name+"Add1"))



  def set_weights(self,vit):
     
    t = self.transformer_layers
    wvit = np.load(vit)

    for i in range(self.transformer_layers):
     norm0=i
     norm1=i+t
     attention = i+2*t
     dense0 = i+3*t
     dense1 = i+4*t
     dropout0 = i+5*t
     dropout1 = i+6*t
     add0 = i+7*t
     add1 =i +8*t
     projection = add1+3
     pos_embedding = add1+5

     w = []      
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias'])

     self.layers[norm0].set_weights(w)
    
     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias'])
  
     self.layers[norm1].set_weights(w)

     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias'])
     self.layers[attention].set_weights(w)
     
     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias'])
  
     self.layers[dense0].set_weights(w)

     w.clear()
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel'])
     w.append(wvit['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias'])
  
     self.layers[dense1].set_weights(w)

     w.clear()
     w.append(wvit['embedding/kernel'])
     w.append(wvit['embedding/bias'])
  
     self.layers[projection].set_weights(w)

     w.clear()
     w.append(wvit['Transformer/posembed_input/pos_embedding'][0,0:self.sequence_length,:])
  
     self.layers[pos_embedding].set_weights(w)



  def call(self, inputs):
 
    reshape = self.InitialReshape(inputs)
    patches = self.InitialDeconv(reshape)
    projection = self.Projection(patches)
    reshaped_projection = self.Reshape(projection)
    embedding = self.PositionEmbedding(reshaped_projection)

    for i in range(self.transformer_layers):
        
        x1 = self.Norm0[i](embedding)
        attention = self.Attention[i](x1,x1)
        x2 = self.Add0[i]([attention,embedding])
        x3 = self.Norm1[i](x2)
        x3 = self.Dense0[i](x3)
        x3 = self.Dropout0[i](x3)
        x3 = self.Dense1[i](x3)
        x3 = self.Dropout1[i](x3)
        embedding = self.Add1[i]([x3,x2])

    x = self.Flatten(embedding)
    x = self.MapDense(x)
    x = self.MapReshape0(x)
    x = self.MapDeconv(x)
    x = self.MapReshape1(x)
    print(x.shape)

    return x

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer

print('ready')  

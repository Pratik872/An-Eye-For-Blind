import tensorflow as tf
from tensorflow.python.keras.models import Model
from keras.preprocessing.text import Tokenizer
import numpy as np
from gtts import gTTS
from IPython import display
from keras.preprocessing import image
import pickle
import playsound

from constants import *

# Loading the models
inception_V3 = tf.keras.models.load_model('InceptionV3_Features_Extractor.h5')

#Loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense =tf.keras.layers.Dense(embed_dim) #build your Dense layer with relu activation
        
    def call(self, features):
        features =  self.dense(features) # extract the features from the image shape: (batch, 8*8, embed_dim)
        features = tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0) #applying relu activation 
        
        return features

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 =  tf.keras.layers.Dense(units)#build your Dense layer
        self.W2 = tf.keras.layers.Dense(units) #build your Dense layer
        self.V = tf.keras.layers.Dense(1)#build your final Dense layer with unit 1
        self.units=units

    def call(self, features, hidden):
        #features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        hidden_with_time_axis =  hidden[:, tf.newaxis] # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        score =tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) # build your score funciton to shape: (batch_size, 8*8, units)
        attention_weights =  tf.keras.activations.softmax(self.V(score), axis=1)  # extract your attention weights with shape: (batch_size, 8*8, 1)
        context_vector =  attention_weights * features #shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1) # reduce the shape to (batch_size, embedding_dim)
        

        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
        

    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed =  self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)  # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output,state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def PreprocessImage(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SHAPE)
    img = image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img,data_format=None)
    final_img = tf.expand_dims(img, 0)
    return final_img

def ExtractFeaturesV3(image_tensor):
    features_tensor = inception_V3(image_tensor)
    features_tensor = tf.reshape(features_tensor, (features_tensor.shape[0], -1, features_tensor.shape[3]))
    return features_tensor

def ExtractFeaturesEncoder(image_vector):
    encoder = Encoder(embedding_dim)
    encoder.load_weights('Encoder_model_weights')
    features = encoder(image_vector)
    return features

def PredictCaptionDecoder(feature_vector):
    result = []
    decoder = Decoder(embedding_dim,units,vocab_size)
    decoder.load_weights('Decoder_model_weights')

    hidden = decoder.init_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, feature_vector, hidden)

        tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def ReturnCaption(caption):
    pred_caption=' '.join(caption).rsplit(' ', 1)[0]

    speech = gTTS(pred_caption, lang = 'en', slow = False) 
    speech.save('voice.mp3')
    audio_file = 'voice.mp3'
    display.display(display.Audio(audio_file, rate=None))
    
    return pred_caption,audio_file

def SpeakOutCaption(file):
    playsound.playsound(file)




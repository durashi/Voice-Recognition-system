import numpy as np
from keras.callbacks import Callback,CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.models import clone_model, Model, Sequential
from tqdm import tqdm
import keras.backend as K
from keras import layers
from keras.utils import Sequence,plot_model,to_categorical
from tqdm import tqdm
import os
from keras.optimizers import Adam
import multiprocessing
from keras.layers import Dense
from numpy import newaxis

def get_baseline_convolutional_encoder(filters, embedding_dimension, input_shape=None, dropout=0.05):
    encoder = Sequential()

    # Initial conv
    if input_shape is None:
        # In this case we are using the encoder as part of a siamese network and the input shape will be determined
        # automatically based on the input shape of the siamese network
        encoder.add(layers.Conv1D(filters, 32, padding='same', activation='relu'))
    else:
        # In this case we are using the encoder to build a classifier network and the input shape must be defined
        encoder.add(layers.Conv1D(filters, 32, padding='same', activation='relu', input_shape=input_shape))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D(4, 4))

    # Further convs
    encoder.add(layers.Conv1D(2*filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(3 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(4 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.GlobalMaxPool1D())

    encoder.add(layers.Dense(embedding_dimension))

    return encoder


def build_siamese_net(encoder, input_shape,  distance_metric='uniform_euclidean'):
    assert distance_metric in ('uniform_euclidean', 'weighted_euclidean',
                               'uniform_l1', 'weighted_l1',
                               'dot_product', 'cosine_distance')

    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    if distance_metric == 'weighted_l1':
        embedded_distance = layers.Subtract()([encoded_1, encoded_2])
        embedded_distance = layers.Lambda(lambda x: K.abs(x))(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'uniform_euclidean':
        # Still apply a sigmoid activation on the euclidean distance however
        embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
        # Sqrt of sum of squares
        embedded_distance = layers.Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)), name='euclidean_distance'
        )(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'cosine_distance':
        raise NotImplementedError
        # cosine_proximity = layers.Dot(axes=-1, normalize=True)([encoded_1, encoded_2])
        # ones = layers.Input(tensor=K.ones_like(cosine_proximity))
        # cosine_distance = layers.Subtract()([ones, cosine_proximity])
        # output = layers.Dense(1, activation='sigmoid')(cosine_distance)
    else:
        raise NotImplementedError

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese

import soundfile as sf
data1, samplerate = sf.read('testc1.flac')
data2, samplerate = sf.read('testd2.flac')

filters = 128
embedding_dimension = 64
dropout = 0.0
opt = Adam(clipnorm=1.)
LIBRISPEECH_SAMPLING_RATE = 1600
n_seconds = 1
downsampling = 4
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)

siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

from keras.models import load_model
#model=load_model('siamese__filters_128__embed_64__drop_0.0__pad=True (2).hdf5')
siamese.load_weights('siamese__filters_128__embed_64__drop_0.0__pad=True (2).hdf5')


a=[]
b=[]
for i in range(input_length):
    a.append([data1[i]])

for j in range(input_length):
    b.append([data2[j]])

    
a=np.array(a,ndmin=3)
b=np.array(b,ndmin=3)
##np.reshape(a, a.shape + (1,))
##np.reshape(b, b.shape + (1,))
##array_1=a[:,:,newaxis]
##array_2=b[:,:,newaxis]
##array_1 = [[['2.0346' for col in range(1)] for col in range(input_length)]for row in range(1)]
##array_2 = [[['1.02546' for col in range(1)] for col in range(input_length)]for row in range(1)]
print(1-siamese.predict([a,b]))












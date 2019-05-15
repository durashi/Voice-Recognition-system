import numpy as np
from keras.callbacks import Callback,CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.models import clone_model, Model, Sequential, load_model
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
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf


app =Flask(__name__)

graph = tf.get_default_graph()


def get_baseline_convolutional_encoder(filters, embedding_dimension, input_shape=None, dropout=0.05):
    encoder = Sequential()

    # Initial conv
    if input_shape is None:
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
        
        embedded_distance = layers.Subtract(name='subtract_embeddings')([encoded_1, encoded_2])
        
        embedded_distance = layers.Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)), name='euclidean_distance'
        )(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'cosine_distance':
        raise NotImplementedError
    else:
        raise NotImplementedError

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese


##optional definition
def get_model():
    global model
    model = load_model('siamese__filters_128__embed_64__drop_0.0__pad=True (2).hdf5')
    print ("model loaded")


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
siamese.load_weights('siamese__filters_128__embed_64__drop_0.0__pad=True (2).hdf5')


print("i m running")


def preprocess(data):
    a=[]
    for i in range(input_length):
        a.append([data[i][0]])
    a=np.array(a,ndmin=3)
    return a

def predict(a,b):
    return (1-siamese.predict([a,b]))


@app.route('/predict', methods=["GET","POST"])
def pre():
    predict_value = 0
    if request.method == 'POST':
        #messege = request.files.getlist("file[]")
        print("000000000000000000000000000000000000000000000")

        f1 = request.files['file_1']
        f1.save(secure_filename(f1.filename))   
        f2 = request.files['file_2']
        f2.save(secure_filename(f2.filename))
        data1, samplerate = sf.read(f1.filename)
        data2, samplerate = sf.read(f2.filename)
        pa1 = preprocess(data1)
        pa2 = preprocess(data2)
        global graph
        with graph.as_default():
            prediction = predict(pa1,pa2)
            predict_value = prediction[0][0]
        print(prediction)
    return render_template("predict.html", prediction = predict_value)

@app.route('/')
def index():
    return render_template('predict.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')




if __name__=='__main__':
    app.run(port="5000",debug=True)

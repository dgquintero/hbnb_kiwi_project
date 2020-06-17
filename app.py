#!/usr/bin/env python
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Tensor flow
import tensorflow as tf
from tensorflow import Graph, Session
# Keras
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import glorot_uniform
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
global graph,model
graph = tf.get_default_graph()

# Define a flask app
app = Flask(__name__)

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        MODELS_PATH = 'models/model_new.h5'
        WEIGHT_MODELS = 'models/weights_new.h5'
        cnn = tf.keras.models.load_model(MODELS_PATH)
        cnn.load_weights(WEIGHT_MODELS)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, cnn):

    x = load_img(img_path, target_size=(100, 100))

    # Preprocessing the image
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    with model_graph.as_default():
        with tf_session.as_default():
            array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, cnn)

        if preds == 0:
            return ("Close")
        elif  preds == 1:
            return ("Medium")
        elif preds == 2:
            return ("Far")
        elif preds == 3:
            return ("Super far")  
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)


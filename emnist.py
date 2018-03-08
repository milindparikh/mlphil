"""
   Copyright 2018 Milind Parikh

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""



# Mute tensorflow debugging information on console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template, jsonify
from scipy.misc import imsave, imread, imresize
from scipy.io import loadmat
import numpy as np
import argparse
from keras.models import model_from_yaml
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils

import re
import base64
import pickle

app = Flask(__name__)

def write_data(instrs):
    pass

def load_data (instrs):
    """ Loads data from a mat file 
    
        '--file' parameter 
        '--height' parameter
        '--width' parameter
        
        The return value is the training and test data along with their labels & mapping
    """
    
    def conv_data(tot_data):
        tot_data = tot_data.astype(np.float32)
        tot_data /= 255

        return tot_data
    
    
    emnist = loadmat(instrs.get('file', 'matlab/emnist-digits.mat'))

    max_ = len (emnist["dataset"][0][0][0][0][0][0])
    train_data = emnist["dataset"][0][0][0][0][0][0][:max_].reshape(max_, int( instrs.get('height', 28)), int( instrs.get('width', 28)), 1)
    train_data = conv_data(train_data)
    train_labels  = emnist["dataset"][0][0][0][0][0][1]

    max_ = len (emnist["dataset"][0][0][1][0][0][0])
    test_data = emnist["dataset"][0][0][1][0][0][0][:max_].reshape(max_, int( instrs.get('height', 28)), int( instrs.get('width', 28)), 1)
    test_data = conv_data(test_data)
    test_labels = emnist["dataset"][0][0][1][0][0][1]
        
        
    mapping = {kv[0]: kv[1:][0] for kv in emnist["dataset"][0][0][2]}
        
    return (train_data, train_labels, test_data, test_labels, mapping)

    

def write_model(model, mapping, parms) :
    model_yaml = model.to_yaml()
    with open(parms.get("--modelYamlFile", "bin/model.yaml"), "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, parms.get("--modelWeights", "bin/model.h5"))

    pickle.dump(mapping, open(parms.get("--mapping", "bin/mapping.p"), 'wb' ))
    

    
def load_model(parms):
    global _mapping, _model
    
    _mapping = pickle.load(open(parms.get("--mapping", "bin/mapping.p"), 'rb'))

    # load YAML and create model
    yaml_file = open(parms.get("--modelYamlFile", "bin/model.yaml"), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    _model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    _model.load_weights(parms.get("--modelWeights", "bin/model.h5"))
    return _model



def infer (x):
    global _model, _mapping
    # Predict from model
    out = _model.predict(x)

    # Generate response
    return ( {'prediction': chr(_mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]})


def learn (parms):
    
    def model(parms):
        
        height = int (parms.get('--height', 28))
        width = int (parms.get('--width', 28))
        nb_classes = int (parms.get('--nb_classes', 10))
                      
        
        # Initialize data
        
        input_shape = ( height , width, 1)

        # Hyperparameters
        nb_filters = 32 # number of convolutional filters to use
        pool_size = (2, 2) # size of pooling area for max pooling
        kernel_size = (3, 3) # convolution kernel size

        model = Sequential()
        model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
        model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

        print(model.summary())
        return model

    
    def train (model, trainingdata, parms):
        
        
        batch_size = int(parms.get('batch_size', 1024))
        epochs = int(parms.get('epochs', 1))
        nb_classes = int (parms.get('--nb_classes', 10))
        
        (train_data, train_labels, test_data, test_labels, mapping) = trainingdata
        
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        
        model.fit(train_data, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_data, test_labels) )
        
        
        score = model.evaluate(test_data, test_labels, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # Offload model to file
        write_model (model, mapping, parms)
        

    
    trainingdata = load_data(parms)
    
    parms['--nb_classes'] = len (trainingdata[len(trainingdata) -1])
    model = model(parms)
    
    train(model, trainingdata, parms)





@app.route('/ws/emnist/digits/infer/', methods=['GET','POST'])
def ws_emnist_digits_infer():
    def parseImage(imgData, filename):
        # parse canvas bytes and save 
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open(filename,'wb') as output:
            output.write(base64.decodebytes(imgstr))
            
    def transformImage(filename) :
        x = imread(filename, mode='L')
        x = np.invert(x)

        # Visualize new array
        imsave('resized.png', x)
        x = imresize(x,(28,28))

        # reshape image data for use in neural network
        x = x.reshape(1,28,28,1)

        # Convert type to float32
        x = x.astype('float32')

        # Normalize to prevent issues with model
        x /= 255
        return x


    parseImage(request.get_data(), 'output.png')
    x = transformImage('output.png')
    
    return jsonify(infer(x))
    
def run_learn_mode (args):
    learn(args)
    #training_data = load_data(args)
    #    model = build_net(training_data, width=args.width, height=args.height, verbose=args.verbose)
    #    train(model, training_data, epochs=args.epochs)
    #print ( training_data [len(training_data) - 1] )



def run_infer_mode (args) :
    load_model(args)
    app.run(host=args.get("host"), port=args.get("port"))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='A fully self container app for learning and infering models  on the EMNIST dataset')
    parser.add_argument('--mode', type=str, default='learn', help='The mode of this function')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to run the flask server on')
    parser.add_argument('--port', type=int, default=5001, help='The port to run the flask server on')


    args, unknown = parser.parse_known_args()
    args = vars(args)
    
    for i in range (0, len(unknown), 2) :
        uarg = unknown[i]
        uarg = remove_prefix(uarg, "--");
        uarg = remove_prefix(uarg, "-");
        args[uarg] = unknown[i+1]
        
    print (args)
    if (args.get("mode") == "learn"):
        run_learn_mode(args)
    else:
        run_infer_mode(args)

    

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
import ast;


from flask import Flask, request, render_template, jsonify

import argparse

import re
import base64
import pickle
import requests

app = Flask(__name__)


@app.route("/")
def index():
    ''' Render index for user connecting to /
    '''
    return render_template('index.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    r = requests.post(args.wsURL, data=request.get_data())
    print (ast.literal_eval(r.text))
    return jsonify(ast.literal_eval(r.text))

if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='A webapp for testing models generated from training.py on the EMNIST dataset')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to run the flask server on')
    parser.add_argument('--port', type=int, default=5000, help='The port to run the flask server on')
    parser.add_argument('--wsURL', type=str, default='http://localhost:5001/ws/emnist/digits/infer/', help='The webservice to call for real prediction')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)

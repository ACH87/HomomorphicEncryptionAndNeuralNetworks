from flask import Flask, request, abort, jsonify, send_file
from flask_restful import Resource, Api
import recommendation_test
import math
from Security_Functions import PallierFunctions
from keras.applications import (vgg16,  vgg19, xception,
                                inception_v3,  inception_resnet_v2,
                                mobilenet,densenet, nasnet, mobilenet_v2, imagenet_utils)
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import zipfile
import numpy as np
import os
import tensorflow as tf
vgg_model = vgg16.VGG16(weights='imagenet')
import pandas as pd
import json

DATASET_LOC = 'myntradataset/testing/'
files = os.listdir(DATASET_LOC)

image_width = eval(str(vgg_model.layers[0].output.shape[1]))
image_height = eval(str(vgg_model.layers[0].output.shape[2]))

c, db_features = recommendation_test.train()

pictures = []
features = []

for f in files:
        # load and resize image
        img = load_img(DATASET_LOC + f, target_size=(image_width, image_height))

        # turn img into an array of shape (244, 244,3)
        array_img = img_to_array(img)

        # add additional dimension to array_img to fit the dimensions needed by the vgg model
        image = np.expand_dims(array_img, axis=0)
        pictures.append(image)

app = Flask(__name__)
api = Api(app)

pal = PallierFunctions(30,50)

@app.route('/findSimilar', methods=['POST'])
def find_similar():
    if not request.json:
        abort(400)

    individual_features = request.json['individual_features'][0]
    features.append(individual_features)
    sum_sqrt = request.json['sum_sqrt']
    number_of_images = request.json['number_of_images']
    db_features_arr = db_features.tolist()
    similarities = []
    result = {}
    counter = 0
    if request.json['encrypted']:
        for x in db_features_arr:
            numerator = pal.encrypt([[0]])[0][0]
            denominator = 0
            for y in range(len(x)):
                mult = pal.multiplicativeHomomorphishm(int(individual_features[y]), int(x[y]))
                numerator = pal.additiveHomomorphism(numerator, cipher_value=mult)
                denominator += x[y]**2

            similarities.append({'index': counter, 'num': numerator,
                                 'den': pal.multiplicativeHomomorphishm(int(sum_sqrt), int(math.sqrt(denominator)))})
            counter +=1

        for val in similarities:
            result[val['index']] = abs(pal.decrypt([[val['num']]])[0][0] / pal.decrypt([[val['den']]])[0][0])
        result = sorted(result.items(), key=lambda k: k[1], reverse=True)
        print(result)
        sorted_res = result[0:number_of_images]
        response = []
        for x in sorted_res:
            response.append(files[x[0]])
    else:
        for x in db_features_arr:
            cos = recommendation_test.calculate_cos(x, individual_features)
            # know sumsqrt is correct
            similarities.append(cos)
        print(similarities)
        idx = sorted(range(len(similarities)), key=lambda i: similarities[i])[-number_of_images:]
        response = []
        for i in idx:
            response.append(files[i])

    print(response)

    zipf = zipfile.ZipFile('Name.zip', 'w', zipfile.ZIP_DEFLATED)
    for file in response:
        zipf.write(DATASET_LOC+file)
    zipf.close()

    return send_file('Name.zip',
            mimetype='zip',
            attachment_filename='Name.zip',
            as_attachment=True)


@app.route('/public_key', methods=['GET'])
def get_public_key():
    return pal.public_key


@app.route('/get_features', methods=['GET'])
def get_features():
    return jsonify(list = features)

@app.route('/scaled_tanh', methods=['POST'])
def scale_tanh():
    string_tensor = request.json['tensor']
    tf.convert_to_tensor(string_tensor)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888')

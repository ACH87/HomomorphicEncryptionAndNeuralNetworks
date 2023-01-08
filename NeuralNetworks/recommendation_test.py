from keras.applications import (vgg16,  vgg19, xception,
                                inception_v3,  inception_resnet_v2,
                                mobilenet,densenet, nasnet, mobilenet_v2, imagenet_utils)
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Model
from keras.layers import Conv2D
import pandas
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import os
import matplotlib.pyplot as plt
import Security_Functions
import math
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATASET_LOC = 'myntradataset/testing/'
DATASET_LOC_2 = 'myntradataset/images/'

# all the images
files = os.listdir(DATASET_LOC)

vgg_model = vgg16.VGG16(weights='imagenet')
vgg19_model = vgg19.VGG19(weights='imagenet')
# mobv2= mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0,
#                                        include_top=True,
#                                        weights='imagenet')
# nasnetmobile = nasnet.NASNetMobile(weights="imagenet")
# largest_dense_net = densenet.DenseNet201(weights="imagenet")
# mobilenet_ = mobilenet.MobileNet(weights="imagenet")
# incepv2 = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',
#                                                 input_tensor=None, input_shape=None, pooling=None, classes=1000)
# incepv3 = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
#                                    input_shape=None, pooling=None, classes=1000)
# Xception_ = xception.Xception(include_top=True, weights='imagenet', input_tensor=None,
#                               input_shape=None, pooling=None, classes=1000)
# large_nasnet = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet',
#                                   input_tensor=None, pooling=None, classes=1000)

pal = Security_Functions.PallierFunctions(min=100, max=200)

cos_similarities_df = None
feat_extract = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)

def train():
    # using the vgg model to preprocess data
    # need to eval output to get an integer object
    print(str(vgg_model.layers[0]))
    image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    pictures = []

    for f in files:
        # load and resize image
        img = load_img(DATASET_LOC + f, target_size=(image_width, image_height))

        # turn img into an array of shape (244, 244,3)
        array_img = img_to_array(img)

        # add additional dimension to array_img to fit the dimensions needed by the vgg model
        image = numpy.expand_dims(array_img, axis=0)
        pictures.append(image)

    # pictures in an array of numpy arrays, a numpy array for each image, vstacking creates one array
    print('vstacking')
    images = numpy.vstack(pictures)
    print('subracting rgb')

    # subtracts the mean RGB values for the images and retrieve a matrix
    dense_matrix = imagenet_utils.preprocess_input(images)

    print('dense matrix', dense_matrix)

    #TODO here is where the encryption code will go

    print('extracting features')

    # finished preprocessing now needs to be fitted with algorithm

    print('obtaining image features')
    img_features = []
    for x in dense_matrix:
        img_features.append(feat_extract.predict(numpy.reshape(x,[1,224,224,3])))
    img_features = feat_extract.predict(dense_matrix)

    print('img_features', img_features)

    # calculate cosine similarities
    cosSimilarities = cosine_similarity(img_features)

    return pandas.DataFrame(cosSimilarities, columns=files, index=files), img_features


def findFeatures(path):
    image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    # load and resize image
    img = load_img(path, target_size=(image_width, image_height))

    # turn img into an array of shape (244, 244,3)
    array_img = img_to_array(img)

    # add additional dimension to array_img to fit the dimensions needed by the vgg model
    image = numpy.expand_dims(array_img, axis=0)
    # print('vstacking')
    # images = numpy.vstack(image)
    # print('subracting rgb')

    # subtracts the mean RGB values for the images and retrieve a matrix
    dense_matrix = imagenet_utils.preprocess_input(image)

    print('dense matrix', dense_matrix)

    #TODO here is where the encryption code will go

    print('extracting features')

    # finished preprocessing now needs to be fitted with algorithm

    print('obtaining image features')
    img_features = feat_extract.predict(dense_matrix)
    return img_features


def most_similar_to(given_img, nb_closest_images=4):
    print("-----------------------------------------------------------------------")
    print("original manga:")

    # using the vgg model to preprocess data
    # need to eval output to get an integer object
    image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    original = load_img(DATASET_LOC + given_img,
                        target_size=(image_width, image_height))

    plt.imshow(original)
    plt.show()

    # turn img into an array of shape (244, 244,3)
    array_img = img_to_array(original).tolist()

    for row in array_img:
        row = pal.encrypt(message=row)[0]

    # encrypted = array_to_img(pal.encrypt_numpy(array_img.astype('int')))

    plt.imshow(array_img)
    plt.show()

    print("-----------------------------------------------------------------------")
    print("most similar manga:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1]

    for i in range(0, len(closest_imgs)):
        original = load_img(DATASET_LOC + closest_imgs[i],
                        target_size=(image_width, image_height))
        plt.imshow(original)
        plt.show()
        print("similarity score : ", closest_imgs_scores[i])

def calculate_cos(pred, comp):
    num = 0
    pred_sumrt = 0
    comp_sumrt = 0
    if len(pred) != len(comp):
        raise Exception("two features are not the same length, features 1 {} feature 2 {}".format(len(pred), len(comp)))
    for x in range(len(pred)):
        num += (pred[x] * comp[x])
        pred_sumrt += pred[x]**2
        comp_sumrt += comp[x]**2

    den = math.sqrt(pred_sumrt) * math.sqrt(comp_sumrt)
    return num/den

def calculate_cos_enc(pred, comp):
    num = 1
    pred_sumrt = 0
    comp_sumrt = 0
    if len(pred) != len(comp):
        raise Exception("two features are not the same length, features 1 {} feature 2 {}".format(len(pred), len(comp)))
    for x in range(len(pred)):
        if comp[x] != 0:
            if num == 0:
                num = 1
            num *= pow(int(comp[x]), int(pred[x]), pal.public_key['n']**2)
            num %= pal.public_key['n']**2
        pred_sumrt += pred[x] ** 2
        comp_sumrt += comp[x] ** 2
    enc = pal.encrypt([[math.sqrt(comp_sumrt)%pal.public_key['n']]])[0][0]
    den = pow(enc, int(math.sqrt(pred_sumrt)%pal.public_key['n']) , pal.public_key['n']**2)
    # den = pred_sumrt * comp_sumrt
    print('num', num)
    print('den', den)
    return pal.decrypt_1d(num)
    # return pal.decrypt_1d(num)**2 / pal.decrypt_1d(den)
    #50% accurate return pal.decrypt_1d(num) / den**0.5


def most_similar_img_enc(img_features, features, nb_closest_images=4):
    result = []
    print('features', features)
    for img in img_features:
        cos = calculate_cos_enc(img, features)
        result.append(cos)

    print('result', result)

    idx = sorted(range(len(result)), key=lambda i: result[i])[-nb_closest_images:]

    image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    for i in idx:
        print(result[i])
        original = load_img(DATASET_LOC + files[i], target_size=(image_width, image_height))
        plt.imshow(original)
        plt.show()

    return idx

def most_similar_img_features(img_features, features, nb_closest_images=4):
    result = []
    print('features', features)
    for img in img_features:
        cos = calculate_cos(img, features)
        result.append(cos-1)

    idx = sorted(range(len(result)), key=lambda i: result[i])[-nb_closest_images:]
    print(idx)
    return idx

    # using the vgg model to preprocess data
    # need to eval output to get an integer object
    # image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    # image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    # for i in idx:
    #     original = load_img(DATASET_LOC + files[i], target_size=(image_width, image_height))
    #     plt.imshow(original)
    #     plt.show()



"""
method to test if we can encrypt then perform homomorphic encryption before decrypting to get initial image
"""
def testHomomorphicImageEnc(given_img, nb_closest_images=4):
    pal = Security_Functions.PallierFunctions(15, 25)
    print("-----------------------------------------------------------------------")
    print("original manga:")

    # using the vgg model to preprocess data
    # need to eval output to get an integer object
    image_width = eval(str(vgg_model.layers[0].output.shape[1]))
    image_height = eval(str(vgg_model.layers[0].output.shape[2]))

    original = load_img(DATASET_LOC + given_img,
                        target_size=(image_width, image_height))

    plt.imshow(original)
    plt.show()

    # turn img into an array of shape (244, 244,3)
    array_img = img_to_array(original)

    # for row in array_img:
    #     counter = 0
    #     for column in row:
    #         row[counter] = elgamal.encrypt(msg=column)[0]
    #         counter +=1
    encrypted = []
    for row in range(len(array_img)):
        encrypted.append(pal.encrypt(array_img[row]))

    addition = pal.encrypt([[1]])

    encrypted_img = array_to_img(encrypted)

    plt.imshow(encrypted_img)
    plt.show()

    for x in encrypted:
        for y in x:
            for z in range(len(y)):
                y[z] = pal.additiveHomomorphism(y[z], cipher_value=addition[0][0])

    decrypted = []
    for row in range(len(encrypted)):
        decrypted.append(pal.decrypt(encrypted[row], convoluted_image=True))

    plt.imshow(array_to_img(decrypted))
    plt.show()

    print("-----------------------------------------------------------------------")
    print("most similar manga:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1]

    for i in range(0, len(closest_imgs)):
        original = load_img(DATASET_LOC + closest_imgs[i],
                            target_size=(image_width, image_height))
        plt.imshow(original)
        plt.show()
        print("similarity score : ", closest_imgs_scores[i])


# cos_similarities_df, img_features = train()
# print('files', files)
# print(cos_similarities_df)
#
# def get_features():
#     return img_features*100
# print(str(type(get_features())))
most_similar_to(given_img='1548.jpg')
# features = findFeatures(DATASET_LOC + '1563.jpg')[0]
# most_similar_img_features(img_features.tolist(),features)
# enc_features = pal.encrypt([((features)%pal.public_key['n']).tolist()])[0]
# most_similar_img_enc(img_features, enc_features)
# testHomomorphicImageEnc(given_img='1163.jpg')
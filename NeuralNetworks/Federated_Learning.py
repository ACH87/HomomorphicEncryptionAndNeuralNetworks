from keras.models import Sequential, load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation
import numpy as np
import math
from Security_Functions import PallierFunctions

pal = PallierFunctions(min=100, max=200)
model = Sequential()
model.add(Conv2D(6, (3,3), input_shape=(32,32,3), padding='same'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(14,14,6)))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(10))
# model.add(Activation(softmax, name='soft'))
# model.add(Activation('softsign'))
model.add(Activation('softmax'))

"""
train with cifar
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def lm(path, X_train=None, y_train=None, epochs=1, batch_size=1, X_test=None, y_test=None):

    return load_model(path)

def predict(path):
    img = load_img(path, target_size=(28, 28))

    array_img = img_to_array(img)
    image = np.expand_dims(array_img, axis=0)
    dense_matrix = imagenet_utils.preprocess_input(image)
    return model.predict(dense_matrix)

def load_dataset(dataset):
    return model.predict(dataset)

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

def find_most_similar(features, number_of_result = 100, sqrt=0.1, encrypted = True):
    scores = {}
    counter = 0

    for x in dataset:
        if encrypted:
            scores[counter] = pal.calculate_cos(features, x, sqrt)
        else:
            scores[counter] = calculate_cos_norm(features, x, sqrt)
        counter+=1

    if encrypted:
        sort = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    else:
        sort = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    result = []
    x= 0
    for res in sort:
        result.append({res: sort[res]})
        x +=1
        if x >= number_of_result:
            break

    return result

def check_accuracy(result, og_class):
    number_of_correct = 0
    print('class', og_class)
    for dic in result:
        print(y_train.tolist()[list(dic.keys())[0]])
        if y_train.tolist()[list(dic.keys())[0]] == og_class:
            number_of_correct +=1

    return number_of_correct/len(result)

def calculate_cos_norm(features, db_features, sqrt):
    num = 0
    dataset_sqrt = 0
    for x in range(len(db_features)):
        num += (features[x] * db_features[x])
        dataset_sqrt += db_features[x]**2
    den = sqrt* math.sqrt(dataset_sqrt)
    return num / den

def check_accuracy_random(pred, y_test):
    result = 0
    for x in range(len(pred)):
        # ft = pal.encrypt_numpy((model.predict(np.reshape(x_test[x], [1,32,32,3]))*1000).astype('int64'))
        ms = find_most_similar(pred[x], number_of_result=100)
        result += check_accuracy(ms, y_test[x].tolist())
    return result/len(pred)

def check_accuracy_select(x_test, y_test):
    result = 0.0
    counter = 0
    for dic in x_test:
        print(y_train.tolist()[list(dic.keys())[0]])
        ms = find_most_similar(dic[counter], number_of_result=100)
        result += check_accuracy(ms, y_test.tolist()[list(dic.keys())[0]])
        counter +=1

    return result / len(x_test)

def get_all_accurate_prediction(data, test, number):
    result = {}
    for x in range(len(data)):
        if np.argmax(data[x]) == np.argmax(test[x]):
            result[x] = data[x]
        if len(result) == number :
            break

    return result


model = lm(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\src\Lenet\lenet_cifar2.h5')
dataset =load_dataset(x_train[:5000])*100
# model.fit(x_train[500:7500], y_train[500:750], epochs=3, batch_size=1)
features = (model.predict(np.reshape(x_test[3], [1,32,32,3]))[0]*100)
print(features)
ft_sqrt = 0
for x in features:
    ft_sqrt += x ** 2

ft_sqrt = math.sqrt(ft_sqrt)

"""
Find most similar not encryted
"""
most_similar = find_most_similar(features.astype('int64'), 100, ft_sqrt, False)
print(most_similar)
print('accuracy not encrypted', check_accuracy(most_similar, y_test[1].tolist()))

# """
# Find most similar encrpted
# """
# pred = pal.encrypt_numpy(features.astype('int64')).tolist()
# print(pred)
# most_similar = find_most_similar(pred, number_of_result=100,sqrt= ft_sqrt)
# print(most_similar)
# print(y_test.tolist()[0])
#
# """
# check accuracy
# """
# accuracy = check_accuracy(most_similar, y_test.tolist()[3])
# print(accuracy)
#
# ov_accuracy = check_accuracy_random([pred], y_test[:1])
# print(ov_accuracy)
#
# accurate_data = get_all_accurate_prediction(model.predict(x_test[:5000]), y_test, 100)
# print(accurate_data)


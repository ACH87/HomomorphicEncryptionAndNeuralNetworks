from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation, Reshape
from keras_layers.Paillier_Convolution import PaillierConvolution
from keras_layers.Paillier_Flatten import Paillier_Flatten
from keras_layers.Paillier_Decrypt import Paillier_Dec
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras_layers.Paillier_Dense import Paillier_Dense
from keras_layers.Paillier_Pooling import Paillier_Pooling
from keras.datasets import fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.optimizers import rmsprop, Adam, SGD
from keras.constraints import min_max_norm
from keras_layers.Activation_Functions import softmax, mod, decrypt
from Security_Functions import PallierFunctions
import datetime
import numpy as np
import os
import keras
from keras.utils.generic_utils import get_custom_objects

import time

get_custom_objects().update({'soft': Activation(softmax)})
# get_custom_objects().update({'mod': Activation(mod)})
pal = PallierFunctions(0, 0, keys={'public_key': {'n': 713, 'g': 16}, 'private_key': {'lambda': 330, 'mu': 139}})

enc_zero = pal.encrypt([[0]])[0][0]

print(enc_zero)

model = Sequential()
model.add(PaillierConvolution(1, (3,3), input_shape=(32,32,3), padding='same', n=713, use_bias=False, kernel_constraint=min_max_norm(0.0,3.0), zero=enc_zero))
model.add(Paillier_Pooling(pool_size=(2,2), padding='same', strides=2, n=713, trainable=False))
model.add(Flatten())
model.add(Paillier_Dense(10, n=713, use_bias=False, zero = enc_zero,kernel_constraint=min_max_norm(0.0,3.0)))
model.add(Paillier_Dec(pal=pal, trainable=False))
model.add(Activation('softmax'))
opt = SGD(0.0000001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

"""
train with cifar
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]


def one_hot_to_label(one_hot):
    return np.argmax(one_hot)

files_train = os.listdir(r'C:/Users/saqla/Documents/Uni/Fourth Year/FYP/recommendation algorithm/src/data/train/')

train = []

for f in files_train:
    # load and resize image
    img = load_img(r'C:/Users/saqla/Documents/Uni/Fourth Year/FYP/recommendation algorithm/src/data/train/'  +  f, target_size=(32, 32))

    # turn img into an array of shape (244, 244,3)
    array_img = img_to_array(img)

    # add additional dimension to array_img to fit the dimensions needed by the vgg model
    image = np.expand_dims(array_img, axis=0)
    train.append(image)

files_test = os.listdir(r'C:/Users/saqla/Documents/Uni/Fourth Year/FYP/recommendation algorithm/src/data/test')

test = []

for f in files_test:
    # load and resize image
    img = load_img(r'C:/Users/saqla/Documents/Uni/Fourth Year/FYP/recommendation algorithm/src/data/test/' + f, target_size=(32, 32))

    # turn img into an array of shape (244, 244,3)
    array_img = img_to_array(img)

    # add additional dimension to array_img to fit the dimensions needed by the vgg model
    test.append(array_img)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# train = np.asarray(train).astype('float32')
# test = np.asarray(test).astype('float32')

print('fitting')
print(np.asarray(train).shape)
start = time.process_time()
hist = model.fit(
    np.asarray(train).reshape([5000,32,32,3]), y_train[:5000],

              batch_size=10,
              epochs=10,
              validation_data=(np.asarray(test).reshape([1000,32,32,3]), y_test[:1000]),
              shuffle=True)
print('time training started: ', datetime.datetime.now())

# model.load_weights(r'C:\Users\saqla\Documents\Uni\Fourth Year\FYP\recommendation algorithm\src\Lenet\lenet_cifar3.h5')

predictions = model.predict(np.reshape(x_test[1], [1,32,32,3]))

print(predictions)

print('target', y_test[0])
def fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=64,
              epochs=1,
              shuffle=True):
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=validation_data,
                  shuffle=shuffle)
print('time taken in seconds', time.process_time() - start)

def save(name='lenet_cifar.h5'):
    model.save('lenet_cifar.h5')



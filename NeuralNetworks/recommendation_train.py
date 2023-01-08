import keras, os
from keras.models import load_model
import numpy as np
import datetime
from keras.datasets import cifar10

model = load_model("vgg16_1.h5")

"""
Pre-process data
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def one_hot_to_label(one_hot):
    return np.argmax(one_hot)

print('time training started: ', datetime.datetime.now())

hist = model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_data=(x_test, y_test),
              shuffle=True)

model.save('vgg16_1.h5')
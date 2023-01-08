from keras.datasets import fashion_mnist, cifar100
from keras.utils import to_categorical
import datetime
import numpy as np
import keras

"""
fitting algorithm
"""
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# normalising data
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(6, (3,3), input_shape=(32,32,3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(16, (3,3), input_shape=(14,14,6), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


"""
train with cifar
"""
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def one_hot_to_label(one_hot):
    return np.argmax(one_hot)

print('time training started: ', datetime.datetime.now())

model.fit(x_train, y_train,
              batch_size=100,
              epochs=1,
              validation_data=(x_test, y_test),
              shuffle=True)

predictions = model.predict(x_test)

print(predictions)

# model.save('lenet_cifar.h5')


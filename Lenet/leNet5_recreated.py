from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation
from keras.datasets import fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.constraints import non_neg
from keras.optimizers import rmsprop, Adam
import datetime
import numpy as np
import keras
from keras_layers.Activation_Functions import square
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'square': Activation(square)})

#TODO TRY KERAS ROUNDING
#TODO TRYING ACTIVATION THAT REMOVES ALL ELEMENTS THAT ARE GREATER THAN 713

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

model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(32,32,3), padding='same', kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Conv2D(6, (3,3), input_shape=(32,32,3), padding='same', kernel_constraint=non_neg(), bias_constraint=non_neg()))
model.add(Activation(square))
# model.add(Activation('relu'))
# model.add(AveragePooling2D(pool_size=(2,2), strides=2))
# model.add(Conv2D(16, (3,3), padding='same',kernel_constraint=non_neg(),use_bias=False))
# model.add(Conv2D(16, (3,3), activation='relu', padding='same',kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Activation(square))
# model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='same'))
# model.add(Conv2D(1, (3,3), activation='relu'))
# model.add(Conv2D(32, (3,3), activation='relu',kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Conv2D(32, (3,3), activation='relu',kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(120,kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Activation('relu'))
# model.add(Activation(square))
model.add(Dense(84,kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Activation(square))
# model.add(Activation('relu'))
model.add(Dense(10,kernel_constraint=non_neg(), bias_constraint=non_neg()))
# model.add(Activation('relu'))
# model.add(Activation(softmax, name='soft'))
# model.add(Activation('softsign'))
# model.add(Activation('softmax'))
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# minst_fit = model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test), shuffle=True)
#
# model.save("lenet_minst.h5")

model.summary()

"""
train with cifar
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

values = {

    "airplane": 0, "automobile": 0, "bird": 0, "cat": 0, "deer": 0, "dog": 0, "frog": 0,
    "horse": 0, "ship": 0, "truck": 0
}

# for x in y_test[4000:5000]:
#     values[classes[x[0]]] = values[classes[x[0]]] + 1
# print(values)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

def one_hot_to_label(one_hot):
    return np.argmax(one_hot)

print('time training started: ', datetime.datetime.now())

print(x_train.shape)
print(y_train.shape)

# hist = model.fit(x_train[:4000], y_train[:4000],
#               batch_size=10,
#               epochs=10,
#               shuffle=True,
#               validation_data=(x_train[4000:5000], y_test[4000:5000]))

model.load_weights('lenet_cifar_non_neg__square_simple.h5')


weights= model.get_weights()
print(weights)

accuracy = model.evaluate(x_test, y_test, batch_size=30)

print(accuracy)
# print(y_test[1].tolist())

# print(model.get_weights())

model.save('lenet_cifar_non_neg__square_simple.h5')


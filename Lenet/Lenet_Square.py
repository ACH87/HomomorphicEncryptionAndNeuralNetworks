from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation, Reshape, Lambda
from keras_layers.Paillier_Convolution import PaillierConvolution
from keras_layers.Paillier_Flatten import Paillier_Flatten
from keras_layers.Paillier_Dense import Paillier_Dense
from keras_layers.Paillier_Decrypt import Paillier_Dec
from keras_layers.Paillier_Pooling import Paillier_Pooling
from keras_layers.Activation_Functions import square, scaled_tanh_pal, softmax
from keras.datasets import fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.optimizers import rmsprop, Adam
import datetime
import numpy as np
import keras
from keras.constraints import min_max_norm, non_neg
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'soft': Activation(softmax)})

"""
fitting algorithm
"""
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# normalising data
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

get_custom_objects().update({'soft': Activation(softmax)})

model = Sequential()
# model.add(PaillierConvolution(3, (3,3), input_shape=(32,32,3), padding='same', kernel_constraint=min_max_norm(1.0,2.0), n=713))
# model.add(Activation('relu'))
# model.add(Paillier_Pooling(pool_size=(2,2), strides=2,n=713))
# model.add(PaillierConvolution(16, (3,3), padding='same'))
# model.add(Activation('relu'))
# model.add(Paillier_Pooling(pool_size=(2,2), strides=2, n=10000000))
model.add(Flatten(input_shape=(32,32,3)))
# model.add(Dense(120))
# model.add(Activation('relu'))
# # model.add(Dense(84))
# # model.add(Activation('relu'))
# model.add(Dense(10, kernel_constraint=non_neg()))
# model.add(Paillier_Dec())
# model.add(Paillier_Dense(10))
model.add((Dense(10)))
model.add(Paillier_Dec())
# model.add(Activation('softmax'))
# model.add(Lambda(softmax, input_shape=(32,32,3)))
model.add(Activation(softmax))
# model.add(Activation('softmax'))
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


model.summary()

"""
train with cifar
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

map = {0: 0, 1: 0, 2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

for i in y_train[:5000].tolist():
    map[i[0]]+=1
print((map))

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('xtest', x_test[0].tolist())

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def one_hot_to_label(one_hot):
    return np.argmax(one_hot)

print('time training started: ', datetime.datetime.now())

train = np.zeros((1,32,32,3))
test = np.where(train==0, 2, train)

#TODO test decryption using just softmax function
hist = model.fit(x_train[:5000], y_train[:5000],
              batch_size=3,
              epochs=3,
              validation_data=(x_train[:1], y_train[:1]),
              shuffle=True)


predictions = model.predict(np.reshape(test,[1,32,32,3]))
print(predictions)

# for row in predictions:
#     for val in row:
#         if val > 1:
#             print('bang')
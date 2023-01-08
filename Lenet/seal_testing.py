from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation
from keras.datasets import fashion_mnist, cifar10
from keras.utils import to_categorical
from keras.constraints import non_neg
from keras.optimizers import rmsprop, Adam
import numpy as np
import keras
from keras_layers.Activation_Functions import softmax
from keras.utils.generic_utils import get_custom_objects

def get_model():
    model = Sequential()
    model.add(Conv2D(1, (3, 3), input_shape=(32, 32, 3), padding='same', kernel_constraint=non_neg(),
					 bias_constraint=non_neg()))
    # model.add(Conv2D(6, (3,3), input_shape=(32,32,3), padding='same', kernel_constraint=non_neg(), bias_constraint=non_neg()))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    # model.add(
	# 	Conv2D(16, (3, 3), activation='relu', padding='same', kernel_constraint=non_neg(), bias_constraint=non_neg()))
    # model.add(Conv2D(16, (3,3), activation='relu', padding='same',kernel_constraint=non_neg(), bias_constraint=non_neg()))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))
    # model.add(Conv2D(32, (3,3), activation='relu'))
    # model.add(Conv2D(32, (3,3), activation='relu',kernel_constraint=non_neg(), bias_constraint=non_neg()))
    # model.add(Conv2D(32, (3,3), activation='relu',kernel_constraint=non_neg(), bias_constraint=non_neg()))
    # model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(120, kernel_constraint=non_neg(), bias_constraint=non_neg()))
    model.add(Activation('relu'))
    model.add(Dense(84, kernel_constraint=non_neg(), bias_constraint=non_neg()))
    model.add(Activation('relu'))
    model.add(Dense(10, kernel_constraint=non_neg(), bias_constraint=non_neg()))
    # model.add(Activation(softmax, name='soft'))
    # model.add(Activation('softsign'))
    model.add(Activation('softmax'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def get_layer():
    return ['conv', 'pool', 'conv', 'pool', 'dense', 'dense', 'dense']

def run_function(name, ipt, wght=None, eval=None, zero=0, square=False):

    if name == 'conv':
        print('convolving')
    elif name == 'pool':
        print('pooling')
    elif name == 'dense':
        #flatten array
        if len(np.array(ipt).shape) != 1:
            print('flattening')
            ipt = ipt.flatten()
        print('densing')

    return ipt

#TODO encrypt weights

model = get_model()
model.load_weights('lenet_cifar_non_neg_simple.h5')
weights = model.get_weights()
encrypted_weights = []

for weight in weights:
    #convolution weight [1,1,1,1]
    ws = len(weight.shape)
    if  ws== 4:
        # print('weight 4d', weight)
        encrypted_weights.append(weight)
        #encrypt4d [1,1,1,1]
    #dense weight [1,1]
    elif ws == 2:
        # print('weight 2d', [weight][0])
        encrypted_weights.append(weight)
    #bias encrypt
    elif ws ==1 :
        if max(weight.tolist()) == 0:
            #dont o anything
            encrypted_weights.append(weight)
        else:
            #encrypt 1d
            # print('weight 1d', weight)
            encrypted_weights.append(weight)

#TODO encrypt images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)

def get_class_name(class_index):
    if class_index < 0 or class_index > 9:
        raise ValueError("Class Index must be > 0 and <= 9")
    return classes[class_index]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


for img in x_test[:1]:
    weight_counter = 0
    for layers in get_layer():
        #if not bias
        if layers != 'pool':
            img = run_function(layers, img, encrypted_weights[weight_counter])
            #handle bias
            if max(encrypted_weights[weight_counter+1]) != 0:
                print('add bias')
            weight_counter +=2
        else:
            img= run_function(layers, img)

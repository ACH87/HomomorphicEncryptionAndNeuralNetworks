import requests
import tensorflow as tf
from Security_Functions import PallierFunctions
import keras.backend as K
import math
import keras.activations as act

pal = PallierFunctions(0, 0, keys={'public_key': {'n': 713, 'g': 16}, 'private_key': {'lambda': 330, 'mu': 139}})

def square(x):
    return x**2

def scaled_tanh_pal(x):
    s = tf.strings.as_string(x)
    print(s)
    #TODO fix this issue with casting
    s = K.cast(tf.convert_to_tensor(s), dtype="float32")
    print(s)
    # s = requests.post('192.168.0.17:8888', bytes(x))
    return s

def softmax(c):
    # val = c ** 20 %10 ** 2
    # val = (val - 1) / 713
    # dec = (val * 139) % 713
    # output = tf.constant([])
    # dec = pal.decrypt_tens(c)
    # den = tf.math.reduce_sum(tf.math.exp(x))
    # for num in range(dec.shape[1]):
    #     num = tf.math.exp(tf.constant([num], dtype=tf.float32))
    #     output = tf.concat([output, tf.math.divide(num, den)], axis=0)
    # print('dec', dec)
    # dec = tf.linalg.normalize(
    #     c, ord='euclidean', axis=0, name=None
    # )

    c=tf.cast(c, tf.float32)
    # c= tf.divide(c, 10000)
    c= tf.where(c<=0.001, tf.constant([0.1], dtype=tf.float32),c)
    c= tf.where(c>=10, tf.constant([0.1], dtype=tf.float32),c)
    dec= tf.math.divide(tf.math.exp(c), tf.reduce_sum(tf.math.exp(c), axis=0))

    # print('dec', dec)

    return dec

def mod(x):
    return x % pal.public_key['n']**2

def greaterThanActivation(x):
    return tf.where(x >= 713, tf.constant([713], dtype=tf.float64), x),

def decrypt(x):
    c = tf.where(x < 1, tf.constant([1], dtype=tf.float32), x)
    c = tf.where(c >= 10, tf.constant([1], dtype=tf.float32), c)
    return pal.decrypt_tens(c)
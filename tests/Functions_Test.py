from Security_Functions import PallierFunctions
import unittest
import tensorflow_core as tf
from keras.models import Model
from keras.layers import Input
from keras_layers.Functions import convultion_function, conv_function_pail, average_pooling, average_pooling_3d, \
    average_pooling_3d_pal, paillier_dense_numpy, convolution_tensors, pooling_tensors, paillier_dense
import numpy as np

class Functions_Test(unittest.TestCase):
    def testDotProductImages(self):
        matrix = [[[0,0,0],[0,0,0],[0,0,0]],
                   [[0,0,0], [156,167,163], [155,166,162]],
                    [[0,0,0], [153,164,160], [154,165,161]]
        ]

        kernel = [[[-1,1,0],[-1,0,1], [1,0,1]],
                   [[0,1,0], [1,-1,1], [-1,-1,0]],
                   [[0,1,1], [1,0,-1], [1,-1,1]

        ]]
        expected = [[[0,0,0], [0,0,0], [0,0,0]], [[0,0,0], [308,-498,164], [0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]
        result = convultion_function(matrix, kernel)
        self.assertEqual(expected, result)

    def testEncryptedDot(self):
        matrix = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        kernel = [[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                  [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                  [[1, 0, 0], [1, 0, 0], [1, 0, 0]]]

        pal = PallierFunctions(20,50)

        for row in range(len(matrix)):
            matrix[row] = pal.encrypt(matrix[row])

        dot_enc = conv_function_pail(matrix, kernel, pal)

        dot_enc = pal.decrypt_3d(dot_enc, convoluted_image=True)
        expected = [[[0,0,0], [0,0,0], [0,0,0]], [[0,0,0], [9,0,0], [0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]

        self.assertEqual(expected, dot_enc)

    def testAveragePooling(self):
        matrix = [[1,2,3,4,5], [11,12,13,14,15], [21,22,23,24,25], [31,32,33,34,35], [41,42,43,44,45]]
        result = average_pooling(matrix, pool_size=(3,3), strides=(2,2))
        expected = [[12, 14], [32,34]]
        self.assertEqual(result, expected)

    def testAveragePooling3d(self):
        matrix = [[[5, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]

        result = average_pooling_3d(matrix, pool_size=(2,2), strides=(2,2))
        expected = [[[2.0,1.0,1.0]]]

        self.assertEqual(expected, result)

    def testAveragePoolingEncry(self):
        matrix = [[[5, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]

        pal = PallierFunctions(15,20)

        for row in range(len(matrix)):
            matrix[row] = pal.encrypt(matrix[row])

        result = average_pooling_3d_pal(matrix, pool_size=(2, 2), strides=(2, 2), pal=pal)
        result = pal.decrypt_3d(result)
        expected = [[[8.0, 4.0, 4.0]]]

        self.assertEqual(expected, result)

    def test_dense_numpy(self):
        input = np.array([[1,2,3,4,5,6,7,8,9,10]])
        kernel = np.array([[3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
        output = paillier_dense_numpy(input, kernel)
        expected = [[52254720000.0, 1.89621927936e+17]]
        self.assertEqual(output.tolist(), expected)

    def test_dense_tensor(self):
        input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.float32)
        kernel = tf.constant([[3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                             dtype=tf.float32)
        output = paillier_dense(input, kernel, n=51, batch_size=2)
        expected = [[558.0, 2304.0], [558.0, 2304.0]]
        self.assertEqual(output.numpy().tolist(), expected)

    def test_conv_tens(self):
        matrix = tf.constant([[[[1,1,1], [1,1,1], [1,1,1]],
                  [[1,1,1], [3,3,3], [2,2,2]],
                  [[1,1,1], [2,2,2], [2,2,2]]
                  ]], dtype=tf.float32)

        kernel = tf.constant([[[[1,1], [1,1], [0,0]], [[1,1], [0,0], [1,1]], [[1,1], [0,0], [1,1]]],
                  [[[0,0], [1,1], [0,0]], [[1,1], [1,1], [1,1]], [[1,1], [1,1], [0,0]]],
                  [[[0,0], [1,1], [1,1]], [[1,1], [0,0], [1,0]], [[1,1], [1,1], [1,1]
                   ]]], dtype=tf.float32)
        expected = np.array([[[[0, 0], [0, 0], [0, 0]], [[0, 0], [956, 1728], [0, 0]], [[0, 0], [0, 0], [0, 0]]]], dtype=np.float32)
        result = convolution_tensors(matrix, kernel, n=50)

        self.assertEqual(result.numpy().tolist(), expected.tolist())

    def test_average_tens(self):
        matrix = tf.constant([[[[2, 3], [2, 3], [2, 3]],
                               [[2, 3], [2, 3], [2, 3]],
                               [[2, 3], [2, 3], [2, 3]]],[[[2, 3], [2, 3], [2, 3]],
                               [[2, 3], [2, 3], [2, 3]],
                               [[2, 3], [2, 3], [2, 3]]]], dtype=tf.float32)

        expected = [[[[16.0, 81.0], [16.0, 81.0]],
                    [[16.0, 81.0], [16.0, 81.0]]],[[[16.0, 81.0], [16.0, 81.0]],
                    [[16.0, 81.0], [16.0, 81.0]]],]

        result = pooling_tensors(matrix, (2,2), strides=(1,1), n=1000000, batch_size=2)
        self.assertEqual(result.numpy().tolist(), expected)



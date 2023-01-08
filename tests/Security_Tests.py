from Security_Functions import PallierFunctions, convultion_function, convultion_function_paillier, dot_product_pal, ElgamalFunctions
import random
import tensorflow_core as tf
from keras.models import Sequential
from keras.layers import Activation
import Math_Functions as mf
from keras.utils.generic_utils import get_custom_objects
import unittest
from keras.datasets import cifar10
import numpy as np
from keras.constraints import min_max_norm, non_neg
import keras
from keras.optimizers import Adam
from keras_layers.Activation_Functions import square

class testSecurity_Functions(unittest.TestCase):

    def testGeneratePrimes(self):
        prime = mf.generatePrime(9,12)
        self.assertEqual(11, prime)

    def greatestCommonDivisor(self):
        gcd = mf.gcd(8, 12)
        self.assertEqual(4, gcd)

    def lowestCommonMultiple(self):
        lcm = mf.lcm(8, 12)
        self.assertEqual(2, lcm)

    def modularMultiplicativeInverse(self):
        mmi = mf.mmi(2, 12, 35)
        self.assertEqual(3, mmi)

    def testElgamalEncryptionDecryption(self):
        # create a 28*28 matrix
        matrix = [[random.randint(0,255) for x in range(9)] for y in range(9)]
        el = ElgamalFunctions()
        p, enc_matrix = el.encrypt2d(matrix)
        dec_matrix = el.decrypt2d(p, enc_matrix)
        self.assertEqual(matrix, dec_matrix, "decrypted matrix {} is not equal to {}".format(dec_matrix, matrix))

    def testPaillierDeryption(self):
        # create a 28*28 matrix
        matrix = [[random.randint(0, 255) for x in range(9)] for y in range(9)]
        pal = PallierFunctions(20, 50)
        enc_matrix = pal.encrypt(matrix)
        dec = pal.decrypt(enc_matrix)
        self.assertEqual(dec, matrix, "decrypted matrix {} is not equal to {}".format(dec, matrix))

    def testPaillierDec1d(self):
        num = random.randint(1, 500)
        pal = PallierFunctions(20, 50)
        enc = pal.encrypt([[num]])[0][0]
        dec = pal.decrypt_1d(enc)
        self.assertEqual(num, dec)

    def testPaillierTensDeryption(self):
        matrix = [[random.randint(1, 256) for x in range(9)] for y in range(9)]
        pal = PallierFunctions(20, 50)
        enc_matrix = pal.encrypt(matrix)
        dec = pal.decrypt_tens(tf.constant(enc_matrix, dtype=tf.float32))
        dec_2 = pal.decrypt_numpy(np.array(enc_matrix))
        self.assertEqual(dec_2.tolist(), dec.numpy().tolist(), "decrypted matrix {} is not equal to {}".format(matrix, dec.numpy().tolist()))

    def testPaillierEncryptionNump(self):
        # create a 28*28 matrix
        matrix = [[random.randint(1, 256) for x in range(9)] for y in range(9)]
        pal = PallierFunctions(20, 50)
        num = pal.encrypt_numpy(matrix)
        enc_matrix = pal.decrypt_numpy(num)
        self.assertEqual(enc_matrix.tolist(), matrix, "decrypted matrix {} is not equal to {}".format(enc_matrix.tolist(), matrix))

    def testConvolution(self):
        matrix = [[1,0,1], [1,0,1],[1,0,1]]
        kernel = [[-1,0,1],[-1,0,1],[-1,0,1]]
        conv = convultion_function(matrix, kernel)
        self.assertEqual(conv,[[0,0,0],[0,0,0],[0,0,0]], "convoluted matrix is {}".format(conv))

    def testPaillierMultiplicative(self):
        pal = PallierFunctions(20, 50)
        enc_value = pal.encrypt([[5]])[0][0]
        multiplication = random.randint(1, pal.public_key['n']-1)
        enc_value_mult = [[pal.multiplicativeHomomorphishm(enc_value, multiplication)]]
        self.assertEqual(pal.decrypt(enc_value_mult), [[(5*multiplication)%pal.public_key['n']]],
                         "decrpted value does not equal {}".format(5*multiplication))

    def testPaillierAdditiveNonEncrypted(self):
        pal = PallierFunctions(20, 50)
        enc_value = pal.encrypt([[5]])[0][0]
        addition = random.randint(1, pal.public_key['n'] - 1)
        enc_value_add = [[pal.additiveHomomorphism(enc_value, original_value=addition)]]
        self.assertEqual(pal.decrypt(enc_value_add), [[(5 + addition) % pal.public_key['n']]],
                         "decrpted value does not equal {}".format(5 + addition))

    def testPaillierAdditiveEncrypted(self):
        print('test')
        pal = PallierFunctions(20, 50)
        print('pal')
        enc_value = pal.encrypt([[5]])[0][0]
        print('enc value')
        addition = random.randint(1, pal.public_key['n'] - 1)
        print('add')
        addition_enc = pal.encrypt([[addition]])[0][0]
        print('test')
        enc_value_add = [[pal.additiveHomomorphism(enc_value, cipher_value=addition_enc)]]
        self.assertEqual(pal.decrypt(enc_value_add), [[(5 + addition) % pal.public_key['n']]],
                         "decrpted value does not equal {}".format(5 + addition))

    def testPaillierNegativeMultiplication(self):
        pal = PallierFunctions(10,30)
        enc_og = pal.encrypt([[10]])[0][0]
        enc_addition = pal.encrypt([[5]])[0][0]
        enc_value = pal.multiplicativeHomomorphishm(enc_og, -1, absolute=True)

        self.assertEqual([[-10]], pal.decrypt([[enc_value]]))

    def testPaillierConvolution(self):
        matrix = [[random.randint(0, 255) for x in range(9)] for y in range(9)]
        pal = PallierFunctions(20, 50)
        enc_matrix = pal.encrypt(matrix)

        convul_matrix = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]

        matrix_after_conv = convultion_function(matrix, convul_matrix)

        for x in matrix_after_conv:
            for y in range(len(x)):
                x[y] %= pal.public_key['n']

        enc_conv_matrix = convultion_function_paillier(enc_matrix, convul_matrix, pal)
        decrypted = pal.decrypt(enc_conv_matrix, True)
        self.assertEqual(decrypted, matrix_after_conv, "decrypted matrix {} not equal to {}".format(decrypted, matrix_after_conv))

    def testPaillierSubtraction(self):
        pal = PallierFunctions(20, 50)
        value1 = random.randint(0, int(pal.public_key['n']/2))
        value2 = [[10]]

        enc = pal.additiveHomomorphism(pal.encrypt([[value1]])[0][0], cipher_value=pal.encrypt(value2)[0][0],
                                       subtraction=True)
        self.assertEqual(pal.decrypt([[enc]], subtraction=True),[[value1-value2[0][0]]],
                         "decrypted {} does not equal actual {}".format(pal.decrypt([[enc]], subtraction=True),
                                                                        value1-value2[0][0]))

    def testPaillierNegativeConvolution(self):
        pal = PallierFunctions(10, 20)
        convul_matrix_neg = [[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]]
        new_matrix = [[random.randint(0, 1) for x in range(3)] for y in range(3)]
        convul_matrix_enc_neg = convultion_function_paillier(pal.encrypt(new_matrix), convul_matrix_neg, pal)
        conv_act = convultion_function(new_matrix, convul_matrix_neg)
        conv_dec = pal.decrypt(convul_matrix_enc_neg, True, True)
        self.assertEqual(conv_act, conv_dec, "decrypted {} does not equal".format(conv_act, conv_dec))

    def testPaillierNegativeMultAdd(self):
        pal = PallierFunctions(20, 30)
        enc_og = pal.encrypt([[5]])[0][0]
        enc_addition = pal.encrypt([[10]])[0][0]
        enc_value = pal.additiveHomomorphism(enc_og, cipher_value=enc_addition, subtraction=True)
        enc_value = pal.multiplicativeHomomorphishm(enc_value, -1, absolute=False, absolute2=True)

        self.assertEqual([[-5]], pal.decrypt([[enc_value]]))


# test()
if __name__ == '__main__':
    unittest.main()

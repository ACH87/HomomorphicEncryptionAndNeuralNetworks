from Security_Functions import PallierFunctions, convultion_function, convultion_function_paillier, dot_product_pal, ElgamalFunctions
import random
import unittest

def test():
    """
    elgamal tests with matrices
    """
    # el = ElgamalFunctions()
    # # create a 28*28 matrix
    # matrix = [[random.randint(0,255) for x in range(9)] for y in range(9)]
    # print('og matrix: ', matrix)
    # p, enc_matrix = el.encrypt2d(matrix)
    # print('encrypted matrix: ', enc_matrix)
    #
    # dec_matrix = el.decrypt2d(p, enc_matrix)
    # print('decryption: ', dec_matrix)
    #
    # convul_matrix = [[-1,0,1], [-1,0,1], [-1,0,1]]
    #
    # print('matrix after conv: ', convultion_function(matrix, convul_matrix))
    #
    # p, enc_conv = el.encrypt2d(convul_matrix, p)

    # print('decryption of conv', el.decrypt2d(p, convultion_function(enc_matrix, enc_conv)))

    # print(convultion_fun  ction([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], [[1,0,1],[0,1,0],[1,0,1]]))

    """
    paillier tests with matrices
    """
    # for _ in range(10):
    #     pal = PallierFunctions(16, 50, r=0, g=0)
    #
    #     enc = pal.encrypt([[1]])
    #     print(enc)
    #     dec = pal.decrypt(enc)
    #     if [[1]] != dec:
    #         raise Exception('original message {} does not equal decrypted {} '.format([[1]], dec))
    #
    #     print('multiplication property')
    #     mes2 = [[enc[0][0]**3 % pal.public_key['n']**2]]
    #     mult = pal.decrypt(mes2)[0][0]
    #
    #     if 3 != mult:
    #         raise Exception('original message {} does not equal decrypted {} '.format(3, mult))
    #
    #     print('addidative property')
    #     add = pal.decrypt([[mes2[0][0] * enc[0][0] % pal.public_key['n']**2]])[0][0]
    #     if 4 != add:
    #         raise Exception('original message {} does not equal decrypted {} '.format(4, add))

    """
    Paillier tests with matrices
    """
    pal = PallierFunctions(5, 10, r=0, g=0)
    # create a 28*28 matrix
    matrix = [[random.randint(0,5) for x in range(9)] for y in range(9)]
    print('og matrix: ', matrix)
    enc_matrix = pal.encrypt(matrix)
    print('encrypted matrix: ', enc_matrix)

    dec_matrix = pal.decrypt(enc_matrix)
    print('decryption: ', dec_matrix)

    convul_matrix = [[1,0,1], [1,0,1], [1,0,1]]

    matrix_after_conv = convultion_function(matrix, convul_matrix)

    for x in matrix_after_conv:
        for y in range(len(x)):
            x[y] %= pal.public_key['n']

    print('matrix after conv: ', matrix_after_conv)

    print('decryption of conv', pal.decrypt(convultion_function_paillier(enc_matrix, convul_matrix, pal),
                                         convoluted_image=True))

    enc_conv = pal.encrypt(convul_matrix)

    """
    test a*c + b*d
    """

    a = 2
    b = 3
    c = 2
    d = 4
    e = 1
    f = 1

    print('expected output is ', (a*c + b*d + e*f) % pal.public_key['n'])

    enc_a = pal.encrypt([[a]])
    enc_b = pal.encrypt([[b]])
    enc_e = pal.encrypt([[e]])

    ac = enc_a[0][0]**c % pal.public_key['n']**2
    bd = enc_b[0][0]**d % pal.public_key['n']**2
    ef = enc_e[0][0]**f % pal.public_key['n']**2

    print('value for  ', pal.decrypt([[(ac * bd * ef) % pal.public_key['n']**2]]))
    print('pal dot product', pal.decrypt([[dot_product_pal([[enc_a[0][0], enc_b[0][0]]], [[c], [d]], pal)]]))

    """
    test negatives
    """
    m1 = [[25]]
    m2 = [[10]]
    m3 = [[5]]

    enc_m1 = pal.encrypt(m1)[0][0]
    enc_m1_neg = [[enc_m1**-1 % pal.public_key['n']**2]]
    enc_m2 = pal.encrypt(m2)[0][0]
    enc_m3 = pal.encrypt(m3)[0][0]
    print('enc_m1 + enc_m3:', enc_m1, enc_m2)
    enc_m1m2 = pal.multiplicativeHomomorphishm(enc_m1, -2, True)
    print('m1*m2', pal.decrypt([[enc_m1m2]]))
    # retrieve enc(m1 - m3)
    enc_m1addm2 = pal.additiveHomomorphism(enc_m1, cipher_value=enc_m3, subtraction=True)
    print(pal.decrypt_prime(enc_m1addm2))
    print(pal.decrypt([[enc_m1addm2]], subtraction=True))
    x = pal.decrypt_subtraction(enc_m1, enc_m3)
    print(x)

    """
    test negative convolution
    """
    convul_matrix_neg = [[-1,0,1],[-1,0,1],[-1,0,1]]
    new_matrix = [[random.randint(0,5) for x in range(3)] for y in range(3)]
    print('new matrix:', new_matrix)
    convul_matrix_enc_neg = convultion_function_paillier(pal.encrypt(new_matrix),convul_matrix_neg, pal)
    print('encrypted convulved matrix:', convul_matrix_enc_neg)
    conv_act = convultion_function(new_matrix, convul_matrix_neg)
    print('actual convulved matrix:', conv_act)
    conv_enc = pal.decrypt(convul_matrix_enc_neg, True, True)
    print('decrypted matrix:', conv_enc)
    if conv_act[1][2] is not conv_enc[1][2]:
        raise Exception

test()
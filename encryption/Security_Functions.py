import random
from numpy import allclose
from keras.layers import Lambda
import math
import numpy as np
import tensorflow as tf
import Math_Functions as mf

class Elgamal:

    a = random.randint(2, 10)

    def __init__(self):
        self.q = random.randint(pow(10, 20), pow(10, 50))
        self.g = random.randint(2, self.q)
        self.key = self.gen_key(self.q)
        self.h = self.power(self.g, self.key, self.q)

        print(self.key)

    def gcd(self, a, b):
        if a < b:
            return self.gcd(b, a)
        elif a % b == 0:
            return b;
        else:
            return self.gcd(b, a % b)

            # Generating large random numbers

    def gen_key(self, q):

        key = random.randint(pow(10, 20), q)
        while self.gcd(q, key) != 1:
            key = random.randint(pow(10, 20), q)

        return key

        # Modular exponentiation

    def power(self, a, b, c):
        x = 1
        y = a

        while b > 0:
            if b % 2 == 0:
                x = (x * y) % c;
            y = (y * y) % c
            b = int(b / 2)

        return x % c

        # Asymmetric encryption

    def encrypt(self, msg, q=None, h=None, g=None):

        en_msg = []
        q = q or self.q
        h = h or self.h
        g = g or self.h

        k = self.gen_key(q)  # Private key for sender
        s = self.power(h, k, q)
        p = self.power(g, k, q)

        for i in range(0, len(msg)):
            en_msg.append(msg[i])

        # print("g^k used : ", p)
        # print("g^ak used : ", s)
        for i in range(0, len(en_msg)):
            en_msg[i] = s * (en_msg[i]) % 252

        return en_msg, p

    def encrypt2(self, msg, q, h, g):

        en_msg = []

        k = self.gen_key(q)  # Private key for sender
        s = self.power(h, k, q)
        p = self.power(g, k, q)

        for i in range(0, len(msg)):
            en_msg.append(msg[i])

        print("g^k used : ", p)
        print("g^ak used : ", s)
        for i in range(0, len(en_msg)):
            en_msg[i] = s * ord(en_msg[i])

        return en_msg, p

    def decrypt(self, en_msg, p, key, q):
        dr_msg = []
        h = self.power(p, key, q)
        for i in range(0, len(en_msg)):
            dr_msg.append(chr(int(int(en_msg[i]) / h)))

        return dr_msg

    def decrypt2(self, en_msg, p, key, q):
        dr_msg = []
        h = self.power(p, key, q)
        for i in range(0, len(en_msg)):
            dr_msg.append(chr(int(int(en_msg[i]) / h)))

        return dr_msg

        # Driver code

    def main(self):

        msg = "2"
        print("Original Message :", msg)

        q = random.randint(pow(10, 20), pow(10, 50))
        g = random.randint(2, q)

        key = self.gen_key(q)  # Private key for receiver
        h = self.power(g, key, q)
        print("g used : ", g)
        print("g^a used : ", h)

        en_msg, p = self.encrypt2(msg, q, h, g)
        dr_msg = self.decrypt(en_msg, p, key, q)
        dmsg = ''.join(dr_msg)
        print("Decrypted Message :", dmsg)

        # testing multiplicative reencrpytion

        msg_2 = "1"

        en_msg2, p2 = self.encrypt2(msg_2, q, h, g)
        print(str(en_msg) + ' ' + str(en_msg2))
        msg_m = int(en_msg[0]) * int(en_msg2[0])
        print(msg_m)
        print('final message: ' + str(self.decrypt(str(msg_m), p+p2, key, q)))


class ElgamalFunctions:

    def __init__(self):
        print('creating')
        self.p, self.g, self.y, self.a = self.generateKey(255, 300)
        print('g: ', str(self.g), ' a: ', self.a, ' p: ', self.p)
        self.y = self.g ** self.a % self.p
        print('y ', str(self.y))
        print('getting public key')
        self.public_key = {'p': self.p, 'g': self.g, 'y': self.y}
        self.secret_key = [self.a]

    def generateKey(self, min, max):
        # find a prime number in range
        primes = []
        for i in range(min, max):
            if i > 1:
                for j in range(2, i):
                    if i % j == 0:
                        break
                else:
                    primes.append(i)
        p = primes[random.randint(0, len(primes)-1)]

        # find number g such that all powers cover all values from 1 to p-1
        for i in range(p):
            list = [False] *p
            list[0] = True
            counter = 0
            while counter < p:
                list[i**counter % p] = True
                counter+=1

            if False not in list:
                g = i

        a = random.randint(2, p-2)
        print('generate y')
        y = g **a %p

        return p,g,y,a

    """
    Encrypt message - an array of values
    """
    def encrypt(self, message, r = None):
        print('encrypted')
        d = [None] * len(message)
        if r is None:
            r = random.randint(1, self.p-1)
        c = self.g ** r % self.p
        for i in range(len(message)):
            d[i] = (message[i] * self.y **r) %self.p

        return c, d

    """
    Encrypt message - a 2d array of message
    """
    def encrypt2d(self, message, r=None):
        print('encrypted')
        # make copy of message
        d = [[message[j][i] for i in range(len(message))] for j in range(len(message[0]))]
        if r is None:
            r = random.randint(1, self.p - 1)
        c = self.g ** r % self.p
        for row in d:
            for i in range(len(row)):
                row[i] = (row[i] * self.y ** r) % self.p

        return c, d

    """
    Decrypt an encrypted value
    :param c represents the randomness
    :param d represents the message
    """
    def decrypt(self, c, d):

        result = [None] * len(d)
        for i in range(len(d)):
            result[i] = (c**(self.p - 1 - self.a) * d[i]) % self.p

        return result

    """
    Decrypt an encrypted value
    :param c represents the randomness
    :param d represents the message
    """
    def decrypt2d(self, c, d):

        result = [[d[j][i] for i in range(len(d))] for j in range(len(d[0]))]

        for row in result:
            for i in range(len(row)):
                row[i] = (c ** (self.p - 1 - self.a) * row[i]) % self.p

        return result

    def findCosineSimilarity(self, vector1, vector2, p1=0, p2=0):
        if len(vector1) != len(vector2):
            raise Exception('Cannot have to different length vectors')

        p = p1*p2
        numerator = 0
        for i in range(len(vector1)):
            if not p:
                numerator += (vector1[i] * vector2[i])
            else:
                # TODO here decrypting the squared of initial value, can overcome this by using additive homomorphism
                numerator += self.decrypt(p, [vector1[i] * vector2[i]])[0]

        a = 0
        b = 0
        for i in range(len(vector1)):
            if p:
                a += self.decrypt(p1**2, [vector1[i]**2])[0]
                b += self.decrypt(p2**2, [vector2[i]**2])[0]
            else:
                a += vector1[i]**2
                b += vector2[i]**2

        denominator = (math.sqrt(a) * math.sqrt(b))

        return numerator /denominator


class PallierFunctions():

    def __init__(self, min=0, max=0, r=0, g=0, mu=0, keys=None):
        if not keys:
            self.r = r
            while True:
                self.p = mf.generatePrime(min, max)
                self.q = mf.generatePrime(min, max)
                print('test1')
                if mf.gcd(self.p*self.q, (self.p-1)*(self.q-1)) == 1 and self.p is not self.q:
                    print('test2')
                    break

            self.n = self.p * self.q
            self.l = int(mf.lcm(self.p-1, self.q-1))

            self.mu = mu
            self.g = g
            while not g or not self.mu:
                print('test3')
                # TODO USE TOP LINE
                # g = random.randint(1, self.n ** 2 - 1)
                g = random.randint(1,20)
                if mf.gcd(g, self.n ** 2) == 1:
                    print('test4')
                    self.g = g
                    self.mu = mf.mmi(self.l, g, self.n)

            self.public_key = {'n': self.n, 'g': self.g}
            self.private_key = {'lambda': int(self.l), 'mu': int(self.mu)}

            print('public key: ', self.public_key)
            print('private key: ', self.private_key)
        else:
            self.r = None
            self.public_key = keys['public_key']
            self.private_key = keys['private_key']

    """
    generate prime in range [min, max)
    """
    def generatePrime(self, min, max):
        primes = []
        for i in range(min, max):
            if i > 1:
                for j in range(2, i):
                    if i % j == 0:
                        break
                else:
                    primes.append(i)

        return primes[random.randint(0, len(primes)-1)]

    # """
    # find greatest common divisor of a in b
    # """
    # def gcd(self,a,b):
    #     while b > 0:
    #         a, b = b, a % b
    #
    #     return a
    #
    # """
    # find lowest common multiple of a and b
    # """
    # def lcm(self, a, b):
    #     return a * b / self.gcd(a,b)
    #
    # """
    # check existence of modular multiplicative inverse and calculate mu
    # """
    # def mmi(self, l,g,n):
    #     x = g**l % n**2
    #     l2 = (x-1) // n
    #     return self.inverse(l2, n)
    #
    # """
    # returns the multiplicative inverse
    # """
    # def inverse(self, n, p):
    #     gcd, x, y = self.eed(n, p)
    #     try:
    #         assert (n * x + p * y) % p == gcd
    #     except AssertionError:
    #         return 0
    #
    #     if gcd != 1:
    #         return 0
    #     else:
    #         return x % p
    #
    # """
    # find extended euclidean distance
    # """
    # def eed(self, a, b):
    #     s, old_s = 0, 1
    #     t, old_t = 1, 0
    #     r, old_r = b, a
    #     while r != 0:
    #         quotient = old_r // r
    #         old_r, r = r, old_r - quotient * r
    #         old_s, s = s, old_s - quotient * s
    #         old_t, t = t, old_t - quotient * t
    #
    #     return old_r, old_s, old_t

    """
    encrypt a 2d message 
    """
    def encrypt(self, message):
        print('enc')
        c = [[0 for i in range(len(message[0]))] for j in range(len(message))]
        r = self.r
        if not r:
            while True:
                # above is correct way to find r, however, for the sake of testing we're gonna use the bootom
                r = random.randint(0, self.public_key['n'])
                # self.r = random.randint(0, 20)
                if mf.gcd(r, self.public_key['n']) == 1:
                    break
        for x in range(len(message)):
            for y in range(len(message[x])):
                c[x][y] = (self.public_key['g']**int(message[x][y])
                           * r**self.public_key['n']) \
                          % self.public_key['n']**2
        print('enc')
        return c

    def decrypt_1d(self, c):
        val = c ** 2 % self.public_key['n']**2
        for i in range(self.private_key['lambda']-2):
            val=pow(int(c), self.private_key['lambda'], self.public_key['n']**2)
            # val = c**self.private_key['lambda'] % self.public_key['n']**2
        val = (val - 1) / self.public_key['n']
        m = (val*self.private_key['mu']) % self.public_key['n']
        return m

    """
    decrypt a 2d message
    :param c = ciphertext to decrypt
    """
    def decrypt(self, c, convoluted_image = False, subtraction=False):
        m = [[0 for i in range(len(c[0]))] for j in range(len(c))]
        for x in range(len(c)):
            for y in range(len(c[x])):
                if convoluted_image and (x == 0 or x == len(c)-1 or y == 0 or y == len(c)-1):
                        m[x][y] = 0
                else:
                    val = (((abs(c[x][y])**self.private_key["lambda"]) % self.public_key['n']**2)-1) //\
                          self.public_key['n']
                    m[x][y] = (val * self.private_key['mu']) % self.public_key['n']
                    if subtraction:
                        x1 = m[x][y]
                        result = ((x1 + math.floor(self.public_key['n'] / 2)) % self.public_key['n']) - math.floor(
                            self.public_key['n'] / 2)

                        if result > self.public_key['n']/2:
                            result -= self.public_key['n']

                        m[x][y] = result
                    if c[x][y] < 0:
                        m[x][y] = -m[x][y]

        return m


    def encrypt_numpy(self, message):
        r = self.r

        if not r:
            while True:
                r = random.randint(0, self.public_key['n'])
                if mf.gcd(r, self.public_key['n']) == 1:
                    break

        # convert python function pow to a numpy function, specifying it takes in 3 inputs, and output one value
        modexp = np.frompyfunc(pow, 3, 1)
        # call the function
        p = modexp(self.public_key['g'], message,self.public_key['n']**2)
        # multiply
        pre_mult = np.multiply(p, r**self.public_key['n'])
        # modulus function of n**2
        c = np.mod(pre_mult, self.public_key['n']**2)

        return c

    def encrypt_tens(self, message):
        r = self.r

        if not r:
            while True:
                r = random.randint(0, 20)
                if mf.gcd(r, self.public_key['n']) == 1:
                    break

        g = tf.constant([self.public_key['g']])
        modexp = tf.math.mod(tf.math.multiply(g, g), self.public_key['n'] ** 2)
        c = modexp
        #
        # todo for loop causing all the issues
        for i in range(self.public_key['g'] - 2):
            modexp = tf.math.mod(tf.math.multiply(modexp, c), self.public_key['n'] ** 2)
        # multiply
        print(r)
        pre_mult = tf.math.multiply(modexp, float(r ** self.public_key['n']))
        # modulus function of n**2
        c = tf.math.mod(pre_mult, self.public_key['n'] ** 2)

        return c

    def decrypt_numpy(self, c, convoluted_image = False, subtraction=False):

        # create numpy that does modeular exponention
        mod = np.frompyfunc(pow, 3, 1)

        modexp = mod(c, self.private_key['lambda'], self.public_key['n']**2)

        num = np.floor_divide(
            np.subtract(
                modexp, 1
            ),self.public_key['n'])
        m = np.mod(np.multiply(num, self.private_key['mu']), self.public_key['n'])

        return m

    def decrypt_tens(self, c, convoluted_image=False, subtraction=False):

        c= tf.cast(c, dtype=tf.float64)
        modexp = tf.math.mod(tf.math.multiply(c,c), self.public_key['n']**2)

        for i in range(self.private_key['lambda']-2):
            modexp = tf.math.floormod(tf.math.multiply(modexp,c), self.public_key['n']**2)

        num = tf.math.divide(
            tf.math.subtract(
                modexp, 1
            ),self.public_key['n'])

        m = tf.math.mod(tf.math.multiply(num, self.private_key['mu']), self.public_key['n'])
        return m

    """
    decrypt a 3d message
    :param c = ciphertext to decrypt
    """

    def decrypt_3d(self, c, convoluted_image=False, subtraction=False):
        m = [[[0 for i in range(3)] for j in range(len(c[0]))] for k in range(len(c))]
        for x in range(len(c)):
            for y in range(len(c[x])):
                for z in range(len(c[x][y])):
                    if convoluted_image and (x == 0 or x == len(c) - 1 or y == 0 or y == len(c) - 1):
                        m[x][y] = [0,0,0]
                    else:
                        val = (((abs(c[x][y][z]) ** self.private_key["lambda"]) % self.public_key['n'] ** 2) - 1) // \
                              self.public_key['n']
                        m[x][y][z] = (val * self.private_key['mu']) % self.public_key['n']
                        if subtraction:
                            x1 = m[x][y][z]
                            result = ((x1 + math.floor(self.public_key['n'] / 2)) % self.public_key['n']) - math.floor(
                                self.public_key['n'] / 2)

                            if result > self.public_key['n'] / 2:
                                result -= self.public_key['n']

                            m[x][y][z] = result
                        if c[x][y][z] < 0:
                            m[x][y][z] = -m[x][y][z]

        return m

    """
    Decryption method to take into account negative numbers
    :param pos encrypted version of the positive message
    :param neg encrypted version pf the megative message
    """
    def decrypt_subtraction(self, pos, neg):
        # retrieve x which is normal decryption
        x = self.decrypt([[self.additiveHomomorphism(pos, cipher_value=neg, subtraction=True)]])[0][0]
        result = ((x+math.floor(self.public_key['n']/2)) % self.public_key['n']) - math.floor(self.public_key['n']/2)
        if result > self.public_key['n']/2 :
            result -= self.public_key['n']
        return result

    """
    Decrypt method that takes applies when a the message is an subtracted message
    """
    def decrypt_prime(self, x):
        x = self.decrypt([[x]])[0][0]
        result = ((x + math.floor(self.public_key['n'] / 2)) % self.public_key['n']) - math.floor(self.public_key['n'] / 2)
        if result > 0:
            result -= self.public_key['n']
        return result

    """
    undo decryption substraction when pos - ng < -n/2
    """
    def undo_decryp(self, message):
        mod_inv = self.modinv((12-30+math.floor(self.public_key['n']/2)), self.public_key['n'])
        result = ((message+self.public_key['n']/2) * mod_inv) + math.floor(self.public_key['n']/2)
        return result

    """
    apply additive homomorphism for 2d array
    : param c the part of the cipher
    : param the value to multply
    """
    def additiveHomomorphism(self, c, cipher_value=None, original_value=None, subtraction=False):
        print('test')
        if original_value:
            enc_val = self.encrypt([[original_value]])[0][0]
        else:
            enc_val = cipher_value
        if subtraction:
            enc_val = self.multiplicativeHomomorphishm(enc_val, -1)

        result = (c*enc_val) % self.public_key['n']**2


        return result

    """
    apply multiplicative homomorphism
    :param c is the cipher text
    :param multiplier is the multiplier
    :param absolute if true perform regular homomorphism and produce negative output
    """
    def multiplicativeHomomorphishm(self, c, multiplier, absolute=False, absolute2=False):
        result = c**abs(multiplier) % self.public_key['n']**2
        if multiplier < 0 and not absolute:
            result = pow(self.modinv(abs(c), self.public_key['n']**2), abs(multiplier), self.public_key['n']**2)
        elif multiplier < 0:
            result = -result
        if absolute2:
            result = -result

        return result

    def egcd(self, a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = self.egcd(b % a, a)
            return (g, x - (b // a) * y, y)

    def modinv(self, a, m):
        g, x, y = self.egcd(a, m)
        if g != 1:
            raise Exception('modular inverse does not exist', a, m)
        else:
            return x % m

    """
    calculate cosine similarity between a cipher feature (feature_ct) and a pain feature (feature_p1)
    """
    def calculate_cos(self, feature_ct, feature_pt, ct_sqrt):
        if len(feature_ct) != len(feature_pt):
            raise Exception(
                "two features are not the same length, features 1 {} feature 2 {}".format(len(feature_ct), len(feature_pt)))

        num = 1
        pt_sumrt = 0

        for x in range(len(feature_ct)):
            # if feature is 0 ignore it
            if feature_ct[x] != 0:
                if num == 0:
                    num = 1
                num *= pow(int(feature_ct[x]), int(feature_pt[x]), self.public_key['n']**2)
                num %= self.public_key['n'] ** 2
            pt_sumrt += feature_pt[x]**2

        den = ct_sqrt*math.sqrt(pt_sumrt)
        return self.decrypt_1d(num) # / den


# initial elgamal test
# el = ElgamalFunctions()
# p, en_message = el.encrypt([1,2,3])
# print('encrypted message: ', str(en_message))
# dec_message = el.decrypt(p, en_message)
# print('dec message: ', str(dec_message))
# print('p: ', str(p))
#
# p2, en_message2 = el.encrypt([3,2,3])
# print('encrypted message2: ', str(en_message2))
# dec_message2 = el.decrypt(p2, en_message2)
# print('dec message2: ', str(dec_message2))
# print('p2: ', str(p2))
#
# dec_mul_mult = el.decrypt(p*p2, [en_message[0]*en_message2[0]])
# print('dec message mult: ', str(dec_mul_mult))
#
# p10, en_message10 = el.encrypt([10])
#
# cosineP = p *p2
# cosine = el.findCosineSimilarity([1,2,3], [3,2,3])
#
# print(cosine)
# print(el.findCosineSimilarity(en_message, en_message2, p1=p, p2=p2))
#
# final_enc = []
"""
find the convolution layer output
:param 
    m = initial matrix
    dot = matrix to multiply
"""
def convultion_function(m, c):
    x = len(m)
    y = len(m[0])
    result = [[0 for i in range(x)] for j in range(y)]
    c_x = len(c)
    c_y = len(c[0])
    if c_x is not c_y:
        raise Exception('Convolution matrix must be a square')
    #halfway point of convolution matrix, to avoid edges
    half_point = int(c_x/2)
    for i in range(x):
        for j in range(y):
            if i < half_point or j < half_point or j > y - half_point -1 or i > x - half_point -1:
                result[i][j] = 0
            else:
                result[i][j] = (dot_product(findSubset(m[i-1:i-1+c_y], j-1, j-1+c_x), c))

    return result

"""
find the convolution layer output for encrypted data
:param 
    c = cioher matrix
    m = matrix to multiply
    pal = pallier cryptosystem
"""
def convultion_function_paillier(c, m, pal):
    x = len(c)
    y = len(c[0])
    result = [[0 for i in range(x)] for j in range(y)]
    m_x = len(m)
    m_y = len(m[0])
    if m_x is not m_y:
        raise Exception('Convolution matrix must be a square')
    #halfway point of convolution matrix, to avoid edges
    half_point = int(m_x/2)
    for i in range(x):
        for j in range(y):
            if i < half_point or j < half_point or j > y - half_point -1 or i > x - half_point -1:
                result[i][j] = 0
            else:
                result[i][j] = (dot_product_pal(findSubset(c[i-1:i-1+m_y], j-1, j-1+m_x), m, pal))

    return result


"""
find the dot product
"""
def dot_product(m, c):
    if len(m) is not len(c) or len(m[0]) is not len(c[0]):
        raise Exception("matrices arent same size")
    result = 0
    for x in range(len(m)):
        for y in range(len(m[0])):
            result += m[x][y] * c[y][x]
    return result

"""
find the dot product with pal encrypted matrices
:param c the cipher
:param m the convolution matrix
:param pal the palier function to encrypt messages
"""
def dot_product_pal(c, m, pal):
    #TODO add check to make sure c and m are the write sizes

    result = 1
    for x in range(len(c)):
        for y in range(len(c[0])):
            # c[x][y] ** m[y][x] == enc(m[x][y])**conv[x][y] == enc(m[x][y]*conv[x][y])
            # result == enc(m[x][y]*conv[x][y]) * enc(m[x-1][y-1]*conv[x-1][y-1])
            # == enc(m[x-1][y-1]*conv[x-1][y-1] +m[x][y]*conv[x][y]
            # result *= (c[x][y] ** int(m[y][x]) % pal.public_key['n']**2)
            r = pal.multiplicativeHomomorphishm(c[x][y], int(m[y][x]), True)
            if r < 0:
                r = pal.additiveHomomorphism(result, cipher_value=abs(r), subtraction=True)
            else:
                r = pal.additiveHomomorphism(result, cipher_value=r)
            result = r

    result %= pal.public_key['n']**2

    return result

"""
find subset of matrix (2d array)
:param
end is exclusive
"""
def findSubset(m, start, end):
    result = [[0 for x in range(len(m))] for y in range(end-start)]
    for i in range(len(m)):
        for j in range(start, end):
            result[i][j-start] = m[i][j]

    return result

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
    print('matrix after conv: ', matrix_after_conv)

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
    m1 = [[2]]
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
    print(pal.decrypt([[enc_m1addm2]]))
    x = pal.decrypt_subtraction(enc_m1, enc_m3)
    print(x)

    x = pal.encrypt([[3]])
    print(pal.decrypt(x))

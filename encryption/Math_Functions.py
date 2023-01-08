import random

def generatePrime(min, max):
    primes = []
    for i in range(min, max):
        if i > 1:
            for j in range(2, i):
                if i % j == 0:
                    break
            else:
                primes.append(i)

    return primes[random.randint(0, len(primes) - 1)]

"""
find greatest common divisor
"""
def gcd(a,b):
    while b > 0:
        a, b = b, a % b

    return a

def lcm( a, b):
    return a * b / gcd(a,b)


def mmi( l,g,n):
    x = g**l % n**2
    l2 = (x-1) // n
    return inverse(l2, n)


def inverse( n, p):
    gcd, x, y = eed(n, p)
    try:
        assert (n * x + p * y) % p == gcd
    except AssertionError:
        return 0

    if gcd != 1:
        return 0
    else:
        return x % p

def eed(a, b):
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = b, a
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t

"""
    encrypt a 2d message 
    """
def encrypt(message, public_key, r=None ):

    c = [[0 for i in range(len(message[0]))] for j in range(len(message))]
    if not r:
        while True:
            # above is correct way to find r, however, for the sake of testing we're gonna use the bootom
            r = random.randint(0, public_key['n'])
            # self.r = random.randint(0, 20)
            if gcd(r, public_key['n']) == 1:
                break
    for x in range(len(message)):
        for y in range(len(message[x])):
            c[x][y] = (public_key['g']**int(message[x][y])
                       * r**public_key['n']) \
                      % public_key['n']**2
    return c

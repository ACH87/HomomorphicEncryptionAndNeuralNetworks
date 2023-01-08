import numpy
import tensorflow_core as tf
import math
from keras.layers import Lambda
from sklearn.feature_extraction import image

"""
convolution functions
@:param matrix, original matrix, represents a colored iamge
@:param kernel, kernal to multiply
"""


def conv_function_pail( matrix, kernel, pal = None, **kwargs):

    x = len(matrix)
    y = len(matrix[1])
    result = [[[0 for i in range(3)] for j in range(y)] for k in range(x)]
    c_x = len(kernel)
    c_y = len(kernel[0])
    # halfway point of convolution matrix, to avoid edges
    half_point = int(c_x / 2)
    for i in range(x):
        for j in range(y):
            if i < half_point or j < half_point or j > y - half_point - 1 or i > x - half_point - 1:
                result[i][j] = [0, 0, 0]
            else:
                result[i][j] = (
                    dot_product_pal(findSubset(matrix[i - 1:i - 1 + c_y], j - 1, j - 1 + c_x), kernel,
                                         pal))

    return result


"""
    find the dot product with pal encrypted matrices
    :param c the cipher
    :param m the convolution matrix
    :param pal the palier function to encrypt messages
    """


def dot_product_pal(c, m, pal):
    # TODO add check to make sure c and m are the write sizes

    if len(m) is not len(c) or len(m[0]) is not len(c[0]):
        raise Exception("matrices arent same size")

    result = [1, 1, 1]
    for x in range(len(c)):
        for y in range(len(c[0])):
            for z in range(len(result)):
                r = pal.multiplicativeHomomorphishm(c[x][y][z], int(m[x][y][z]), True)
                if r < 0:
                    r = pal.additiveHomomorphism(result[z], cipher_value=abs(r), subtraction=True)
                else:
                    r = pal.additiveHomomorphism(result[z], cipher_value=r)
                result[z] = r

    for r in result:
        r %= pal.public_key['n'] ** 2

    return result


"""
find subset of matrix (2d array)
:param
 end is exclusive
"""


def findSubset( m, start, end):
    result = [[0 for x in range(len(m))] for y in range(end - start)]
    for i in range(len(m)):
        for j in range(start, end):
            result[i][j - start] = m[i][j]

    return result


"""
find the convolution layer output
:param 
m = initial matrix
dot = matrix to multiply
"""


def convultion_function(m, c):
    x = len(m)
    y = len(m[0])
    result = [[[0 for i in range(3)] for j in range(y)] for k in range(x)]
    c_x = len(c)
    c_y = len(c[0])

    # halfway point of convolution matrix, to avoid edges
    half_point = int(c_x / 2)
    for i in range(x):
        for j in range(y):
            if i < half_point or j < half_point or j > y - half_point - 1 or i > x - half_point - 1:
                result[i][j] = [0, 0, 0]
            else:
                result[i][j] = (dot_product(findSubset(m[i - 1:i - 1 + c_y], j - 1, j - 1 + c_x), c))

    return result


"""
Find convolution of un encrypted messages, used for testing,
should be noted the dot product is not in fact what is performed,
but rather an elementwise multiplication of two matrices followed by a summation
dot product is used for naming conveniaence
"""


def dot_product(m, c):
    if len(m) is not len(c) or len(m[0]) is not len(c[0]):
        raise Exception("matrices arent same size")
    result = [0, 0, 0]
    for x in range(len(m)):
        for y in range(len(m[0])):
            for z in range(len(result)):
                result[z] += m[x][y][z] * c[x][y][z]
    return result


"""
Find average pooling 2d
@:param inputs the matrix you want to find the pool of
"""


def average_pooling(inputs, pool_size, strides, padding=None, data_format=None):

    result = []
    # create new row
    x = 0
    while x <= len(inputs) - pool_size[0]:
        y = 0
        row = []
        while y <= len(inputs[0]) - pool_size[1]:
            r = 0
            for j in range(pool_size[0]):
                for i in range(pool_size[1]):
                    r += inputs[x+j][y + i]
            r /= pool_size[1] * pool_size[0]
            row.append(r)
            y += strides[1]

        result.append(row)

        x += strides[0]

    return result


"""
average pooling 3d
"""


def average_pooling_3d(inputs, pool_size, strides, padding=None, data_format=None):
    result = []
    # create new row
    x = 0
    while x <= len(inputs) - pool_size[0]:
        y = 0
        row = []
        while y <= len(inputs[0]) - pool_size[1]:
            r = [0,0,0]
            for j in range(pool_size[0]):
                for i in range(pool_size[1]):
                    for z in range(3):
                        r[z] += inputs[x + j][y + i][z]

            r = numpy.divide(r, pool_size[0]*pool_size[1]).tolist()
            row.append(r)
            y += strides[1]

        result.append(row)

        x += strides[0]

    return result


"""
average pooling 3d
"""


def average_pooling_3d_pal(inputs, pool_size, strides, pal, padding=None, data_format=None):
    result = []
    # create new row
    x = 0
    while x <= len(inputs) - pool_size[0]:
        y = 0
        row = []
        while y <= len(inputs[0]) - pool_size[1]:
            r = [1,1,1]
            for j in range(pool_size[0]):
                for i in range(pool_size[1]):
                    for z in range(3):
                        r[z] = pal.additiveHomomorphism(r[z], cipher_value=inputs[x + j][y + i][z], subtraction=False)

            for val in range(len(r)):
                r[val] = r[val] % pal.public_key['n']**2

            row.append(r)
            y += strides[1]

        result.append(row)

        x += strides[0]

    return result


"""
Find Paillier convolution between two different tesnors
"""


def convolution_tensors(matrix, kernel, zero=0,pal=None, strides=(1,1),
            padding='same',
            data_format=None,
            dilation_rate=None,
                        n=1,
                        batch_size=1):

    number_of_elements = kernel.shape[0]*kernel.shape[1]*kernel.shape[2]
    # ksize_0 = kernel.shape[0]*kernel.shape[1]
    # extract image patches, will output a shape of (?, length, width, kernel.length*kernel.widhth)
    patches = tf.cast(tf.image.extract_patches(
            images=matrix, sizes=[1, kernel.shape[0], kernel.shape[1], 1], strides=[1, strides[0], strides[1], 1],
            rates=[1,1,1,1], padding='VALID'
        ), dtype=tf.float64)

    if padding == 'valid':
        rows = patches.shape[1]
        col = patches.shape[2]
    else:
        rows = matrix.shape[1]
        col = matrix.shape[2]

    # flatten image patches into a tensor of shape (matrix.length*matrix.width, kernel.length*kernel.widhth
    # more efficient than multiple for loops
    patches = tf.reshape(patches, [batch_size*patches.shape[1]*patches.shape[2], number_of_elements])
    print('patches', patches)

    output = tf.constant([], dtype=tf.float64)

    if padding == 'same':
        pad = tf.zeros([batch_size,(col - math.ceil(kernel.shape[0]/2) + rows )*kernel.shape[3] ])

    for k in range(kernel.shape[3]):
        # if padding is the same add a row of 0's

        # reshape each kernel into a 1 array of size (,number of elements)
        new_k = tf.reshape(kernel[:,:,:,k], [number_of_elements])
        #
        # mod = numpy.frompyfunc(pow, 3, 1)
        #
        # def lambda_pow(x):
        #     return mod(x, new_k, n ** 2)

        # power = tf.py_function(lambda_pow, [tf.where(patches[:, :]==0, tf.constant([1], dtype=tf.float32), patches[:,:])], Tout=tf.float32)
        prod = tf.math.mod(tf.math.pow(tf.where(patches[:, :]<=0, tf.constant([1], dtype=tf.float64), patches[:,:]), tf.cast(new_k, dtype=tf.float64)), n**2)
        # tf.where(tf.is_nan(has_nans), tf.zeros_like(has_nans), has_nans)
        for y in range(int(prod.shape[1]/3)):
            if y == 0:
                cpy = tf.math.mod(tf.math.cumprod(prod[:,y*3:(y+1)*3],axis=1,  reverse=True)[:,0], n**2)
            else:
                cpy = tf.math.mod(tf.math.multiply(cpy,
                                 tf.math.mod(tf.math.cumprod(prod[:, y*3:(y+1)*3],axis=1,  reverse=True)[:,0], n**2)
                                                   ), n**2)

        # prod = tf.reshape(cpy, [batch_size, 1])
        prod = cpy
        output = tf.concat([output, tf.where(prod<=0, tf.constant([1], dtype=tf.float64), prod)], axis=0)

    if padding == 'same':
        output = tf.concat([pad, tf.cast(tf.reshape(output, [batch_size,
                                                     (rows-2)*(col-2)*kernel.shape[3]]), dtype=tf.float32),
                            pad], axis=1)

    # output = tf.math.floormod(output, n**2)

    return tf.reshape(output, [batch_size, rows, col, kernel.shape[3]])




"""
Method to find average pooling using tensors
"""


def pooling_tensors(inputs, pool_size, strides, zero=0, padding=None, data_format=None,n=1, batch_size=1):

    patches =tf.image.extract_patches(
        images=inputs, sizes=[1, pool_size[0], pool_size[1], 1], strides=[1, strides[0], strides[1], 1],
        rates=[1,1,1,1], padding='VALID'
    )
    print('patches', patches)

    patch_r = patches.shape[1]
    patch_c = patches.shape[2]
    size = patches.shape[3]

    print('inputs', inputs)

    # arr = []
    # for x in range(patches.shape[1] * patches.shape[2]):
    #     arr.append([])

    patches = tf.reshape(patches, [batch_size*patches.shape[1]*patches.shape[2], int(patches.shape[3]/inputs.shape[3]), inputs.shape[3]])

    for x in range(inputs.shape[3]):
        #
        # tf.math.mod(tf.math.pow(
        patch = tf.where(patches[:,:,x] <= 0, tf.constant([zero], dtype=tf.float32), patches[:,:,x])
        for y in range(int(patches.shape[1]/2)):
            slce = tf.cast(patch[:, y*2:(y+1)*2], dtype=tf.float64)
            print('slce', slce)
            average = tf.cast(tf.math.floormod(
                tf.math.cumprod(
                    slce,
                    axis=1, reverse=True)[:,0], n**2),dtype=tf.float32)
            #               ,[patches.shape[1]])n**2)
            if y == 0:
                local = tf.reshape(average, [batch_size*patch_r*patch_c,1])
            else:
                print('local before', local)
                local = tf.concat([local, tf.reshape(average, [batch_size*patch_r*patch_c,1])], axis=-1)
                print('local after', local)
                average = tf.cast(tf.math.floormod(tf.math.cumprod(tf.cast(local, dtype=tf.float64), axis=1, reverse=True)[:,0], n**2), dtype=tf.float32)


        if x== 0:
            output = tf.reshape(average, [batch_size*patch_r*patch_c, 1])
        else:
            average = tf.reshape(average, [batch_size*patch_r*patch_c, 1])
            output = tf.concat([output, average], -1)
    # patches = tf.reshape(patches[:, :, :, x], [patches.shape[1] * patches.shape[2], patches.shape[3]])
    # for x in range(patches.shape[0]):
    #     average = tf.reshape(tf.math.pow(tf.math.cumprod(tf.math.add(patches[:,x,:], [1]), reverse=True)[0], [patches.shape[1]]), [-1])
    #     output = tf.concat([output, average], axis=0)
    tf.debugging.check_numerics(
        output, 'nan', name=None
    )

    return tf.reshape(output, [batch_size, patch_r, patch_c, inputs.shape[3]])


def paillier_dense(inputs, kernel, batch_size=1, n=1, section_size=5, zero=0):
    num_of_outputs = kernel.shape[1]
    output = tf.constant([[]], dtype=tf.float32)
    # output = tf.zeros([batch_size, num_of_outputs])

    for i in range(num_of_outputs):
        power = tf.cast(tf.math.mod(
                tf.math.pow(tf.cast(
                    tf.where(inputs<=0.0001, tf.constant([zero], dtype=tf.float32), inputs), dtype=tf.float64), tf.cast(kernel[:,i], dtype=tf.float64)
                            ),n**2 ), dtype=tf.float32)
        local = power

        for j in range(math.ceil(local.shape[1] / section_size)):
            slce =tf.cast(tf.where(local[:, j*section_size:(j+1)*section_size] <= 0, tf.constant([zero], dtype=tf.float32),
                                                  local[:, j*section_size:(j+1)*section_size]), dtype=tf.float64)
                # slce = local[:, j*section_size: (j+1) * section_size]
            result = tf.math.mod(tf.math.cumprod(slce, axis=1,reverse=True)[:, 0], n**2)
            result = tf.reshape(result, [batch_size, 1])
            if j == 0:
                local_cpy = tf.cast(result, dtype=tf.float32)
                # break
            else:
                local_cpy = tf.cast(tf.concat([local_cpy, tf.cast(result, dtype=tf.float32)], axis=1), dtype=tf.float64)
                local_cpy = tf.cast(tf.reshape(tf.math.mod(tf.math.cumprod(local_cpy, axis=1, reverse = True)[:,0], n**2), [batch_size, 1]), dtype=tf.float32)

        if i == 0:
            output = local_cpy
        else:
            output = tf.concat([output, local_cpy], axis=-1)
    output = tf.cast(output, dtype=tf.float32)
    return tf.reshape(output, [batch_size, num_of_outputs])


def paillier_dense_numpy(inputs, kernel):
    num_of_outputs = kernel.shape[1]
    output = numpy.array([])
    for i in range(num_of_outputs):
        output = numpy.concatenate([output, numpy.reshape(numpy.cumprod(numpy.power(inputs[0], kernel[:,i]), dtype=numpy.int64)[inputs.shape[1]-1], [-1])], axis=0)
    return numpy.reshape(output, [1, num_of_outputs])


"""
Method to calculate output shape of a tensor
"""


def calculateOutputShape(image_dimensions, kernel_dimensions, number_of_kernels, padding='same'):
    if padding is not 'same':
        output_shape = (None, image_dimensions[0] - int(kernel_dimensions[0]/2),
                        image_dimensions[1] - int(kernel_dimensions[1]/2), number_of_kernels)
    else:
        output_shape = (None, image_dimensions[0] - int(kernel_dimensions[0]/2),
                        image_dimensions[1] - int(kernel_dimensions[1]/2), number_of_kernels)

    return output_shape


def batch_conversion(imgs, filters, strides, padding, rate = None):
    filters_shape = filters.shape
    # for _ in range(filters_shape[4])
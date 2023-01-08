from keras import backend as K
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils
from keras_layers.Functions import pooling_tensors

class Paillier_Pooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None,n=1,zero=0, **kwargs):
        super(Paillier_Pooling, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.n=n
        self.zero=zero

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs, **kwargs):
        batch_size = K.shape(inputs)[0]
        output = pooling_tensors(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format,
                                 n=self.n,
                                 zero=self.zero,
                                 batch_size=batch_size)
        return output


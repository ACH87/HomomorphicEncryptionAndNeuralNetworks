from keras.layers import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints
from keras_layers.Functions import dot_product_pal
import keras.backend as K
import tensorflow as tf
import numpy as np

class Paillier_Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Example

    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(Paillier_Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '). '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        x = self.batch_flatten(inputs)
        print('ourput of call', x)
        return x

    def batch_flatten(self, x):
        print('x shape', x.shape)
        # x = tf.reshape(x, [x.shape[1], x.shape[2], x.shape[3]])
        x = tf.reshape(
            x, tf.stack([-1, tf.reduce_prod(tf.shape(x)[1:])],
                        name='stack_' + str(np.random.randint(1e4))))
        print('x output shape', x.shape)
        return x

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Paillier_Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

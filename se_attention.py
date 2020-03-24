'''
ref https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se.py
'''
from keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply, add, Conv1D
import keras.backend as K


def squeeze_excite_block(input_feature, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    channel_axis = -1 
    filters = input_feature._keras_shape[channel_axis]

    se = GlobalAveragePooling1D()(input_feature)
    se = Reshape((1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([input_feature, se])
    return x


def spatial_squeeze_excite_block(input_feature):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv1D(1, 1, activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_feature)

    x = multiply([input_feature, se])
    return x


def channel_spatial_squeeze_excite(input_feature, ratio=16):
    """ Create a channel + spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_feature, ratio)
    sse = spatial_squeeze_excite_block(input_feature)

    x = add([cse, sse])
    return x
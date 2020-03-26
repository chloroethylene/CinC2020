import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras.models import Model
from se_attention import squeeze_excite_block, channel_spatial_squeeze_excite
from attention import CBAM_channel_attention, CBAM_position_attention, AttentionWithContext


def attach_attention_module(layer, attention_module):
    if attention_module == 'cse':  # channel wise SE
        layer = squeeze_excite_block(layer)

    if attention_module == 'csse':  # channel and spatial wise SE
        layer = channel_spatial_squeeze_excite(layer)

    if attention_module == 'channel CBAM':  # channel wise CBAM
        layer = CBAM_channel_attention(layer)

    if attention_module == 'channel spatial CBAM':  # channel spatial wise CBAM
        layer = CBAM_channel_attention(layer)
        layer = CBAM_position_attention(layer)

    return layer


def Net(num_classes=9, input_length=15360, nChannels=12, attention_module=None):
    inputs = Input(shape=(input_length, nChannels))

    x = Conv1D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)

    x = attach_attention_module(x, attention_module)

    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)

    x = attach_attention_module(x, attention_module)

    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)

    x = attach_attention_module(x, attention_module)

    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)

    x = attach_attention_module(x, attention_module)

    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same', name='last_conv_layer')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(0.2)(x)

    x = attach_attention_module(x, attention_module)

    max_pooled = GlobalMaxPooling1D()(x)
    avg_pooled = GlobalAveragePooling1D()(x)
    pooled = concatenate([max_pooled, avg_pooled])

    outputs = Dense(num_classes, activation='sigmoid')(pooled)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def Netbeta(inputs, num_classes=9, include_top=True):
    x = Conv1D(12, 3, padding='same')(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv1D(12, 48, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = CuDNNGRU(12, input_shape=(480, 12), return_sequences=True)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    if include_top:
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(num_classes, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    else:
        return x


def multi_input_model(num_classes=9, input_length=15360, nChannels=1):
    inputs = []
    for i in range(12):
        locals()['input' + str(i + 1)] = Input(shape=(input_length, nChannels))
        locals()['x' + str(i + 1)] = Netbeta(inputs=locals()['input' + str(i + 1)], num_classes=num_classes,
                                             include_top=False)
        if i == 0:
            x = locals()['x1']
        else:
            x = concatenate([x, locals()['x' + str(i + 1)]])
        inputs.append(locals()['input' + str(i + 1)])

    x = Dropout(0.5)(x)

    x = GlobalMaxPooling1D()(x)

    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    # model=Netbeta(inputs=Input(shape=(15360,1)))
    # model = merge_model()
    model = Net()
    model.summary()

    '''
    x = Bidirectional(CuDNNGRU(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Bidirectional(CuDNNGRU(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    '''
    '''
    x = Bidirectional(CuDNNGRU(12, return_sequences=True))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    '''
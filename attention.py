import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import *
from keras.models import Model
from keras.activations import sigmoid

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    
    Note: The layer has been tested with Keras 2.0.6
    
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        # 论文中uit=tanh(Ww hit + bw) 此处的x就是hit
        # 因为hit是GRU变换后的隐层状态
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        
        #return weighted_input
        return K.sum(weighted_input, axis=1)
        
        #avg_pool = K.mean(weighted_input, axis=1)
        #max_pool = K.max(weighted_input, axis=1)
        
        #return avg_pool + max_pool
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


    
    
def CBAM_channel_attention(input_feature, ratio=2):
    '''
    follow the work of 2018_ECCV_Convolutional_Block_Attention
    改成适用于时序信号
    '''
   
    channel = input_feature._keras_shape[-1]
 
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling1D()(input_feature)    
    avg_pool = Reshape((1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,channel)
    
    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return multiply([input_feature, cbam_feature])


def CBAM_position_attention(input_feature):
    kernel_size = 7

    channel = input_feature._keras_shape[-1]
   
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv1D(
                filters = 1,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)(concat)
    
    
    return multiply([input_feature, cbam_feature])




if  __name__ == '__main__':
    input_feature = Input(shape=(480, 24))
 

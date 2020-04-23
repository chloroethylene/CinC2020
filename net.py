# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:44:34 2020

@author: FXZ
"""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras.models import Model
#from CBAM import *


def attach_attention_module(layer, attention_module):
    if attention_module == 'cse': # channel wise SE
        layer = squeeze_excite_block(layer)
    
    if attention_module == 'csse': # channel and spatial wise SE
        layer = channel_spatial_squeeze_excite(layer)
        
    if attention_module == 'channel CBAM': # channel wise CBAM
        layer = CBAM_channel_attention(layer)
    
    if attention_module == 'channel spatial CBAM': # channel spatial wise CBAM
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
    x = Dropout(0.2, name='last_Dropout')(x) 
    
    x = attach_attention_module(x, attention_module)
    
    #x = TransformerEncoder(d_model=256, d_inner_hid=512, n_head=3, layers=1, dropout=0.2)(x)
    
    x = GlobalMaxPooling1D()(x)
    
    outputs = Dense(num_classes,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
   

def ensemble_model():
    inputs=Input(shape=(117,1))
    x = Dropout(0.5, [1, 117, 1])(inputs)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(9,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

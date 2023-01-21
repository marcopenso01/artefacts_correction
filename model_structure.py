import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import logging

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)

def activation_block(x):
    x = ReLU()(x)
    return BatchNormalization()(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = Add()([activation_block(x), x0])  # Residual.
    # Pointwise convolution.
    x = Conv2D(filters, kernel_size=1, kernel_initializer='he_normal')(x)
    x = activation_block(x)
    return x
    

def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)
  
  
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def spatial_attention(input_feature):
    kernel_size = 7
    channel = input_feature.get_shape().as_list()[-1]

    avg_pool = tf.reduce_mean(input_feature, [1, 2], keepdims=True)

    max_pool = tf.reduce_max(input_feature, [1, 2], keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):
    channel = input_feature.get_shape().as_list()[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])
  

def DilatedSpatialConv(dspp_input, filters=32):
    out_1 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=1)(dspp_input)))
    out_4 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=4)(dspp_input)))
    out_7 = ReLU()(BatchNormalization()(Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                                               dilation_rate=7)(dspp_input)))

    x = Concatenate(axis=-1)([out_1, out_4, out_7])
    return x
  

def AttUnet2d(input_size1=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)

    # ConvMix layer 1
    x1 = conv_stem(input_model1, filters=256, patch_size=1)
    for _ in range(8):
        x1 = conv_mixer_block(x1, filters=256, kernel_size=9)
    
    # Conv Layer 1
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1d)))
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x12)))
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(concatenate([x1, x12], axis=-1))

    # ConvMix layer 2
    x2 = conv_stem(input_model1, filters=256, patch_size=2)
    for _ in range(8):
        x2 = conv_mixer_block(x2, filters=256, kernel_size=7)
    
    # Conv Layer 2
    x22 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    x22 = conv_mixer_block(x22, filters=n_filt * 2, kernel_size=3)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(concatenate([x2, x22], axis=-1))
    
    # ConvMix layer 3
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(8):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=7)
    
    # COnv layer3
    x32 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    x32 = conv_mixer_block(x32, filters=n_filt * 4, kernel_size=3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(concatenate([x3, x32], axis=-1))

    # layer4
    x4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    x4 = conv_mixer_block(x4, filters=n_filt * 8, kernel_size=3)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)

    # layer5
    x5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    x5 = conv_mixer_block(x5, filters=n_filt * 16, kernel_size=3) 

    up4 = UpSampling2D(size=(2, 2))(x5)
    skip4 = ResPath(x4, length=1)
    conc4 = concatenate([up4, skip4], axis=3)

    conv6 = conv_mixer_block(conc4, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3 = ResPath(x32, length=2)
    conc3 = concatenate([up3, skip3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = ResPath(x22, length=3)
    conc2 = concatenate([up2, skip2], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = ResPath(x12, length=4)
    conc1 = concatenate([up1, skip1], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    conv_out = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=input_model1, outputs=conv_out)
    logging.info('Finish building model')
    return model
  

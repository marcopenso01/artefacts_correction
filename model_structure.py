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
    x = Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def Unet(input_size1=(160, 160, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # layer1 2D
    x1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2 2D
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)))
    conv2 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3 2D
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)))
    conv3 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4 2D
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(pool3)))
    conv4 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # layer5 2D
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)))
    conv5 = ReLU()(BatchNormalization()(Conv2D(n_filt * 16, (3, 3), padding='same', kernel_initializer='he_normal')(conv5)))

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(up4)))
    conv6 = ReLU()(BatchNormalization()(Conv2D(n_filt * 8, (3, 3), padding='same', kernel_initializer='he_normal')(conv6)))

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(up3)))
    conv7 = ReLU()(BatchNormalization()(Conv2D(n_filt * 4, (3, 3), padding='same', kernel_initializer='he_normal')(conv7)))

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(up2)))
    conv8 = ReLU()(BatchNormalization()(Conv2D(n_filt * 2, (3, 3), padding='same', kernel_initializer='he_normal')(conv8)))

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(up1)))
    conv9 = ReLU()(BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)))

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_model1, outputs=output)
    logging.info('Finish building model')

    return model


def ResPath(encoder, length=1):
    '''
    ResPath
    Arguments:
        length {int} -- length of ResPath
        encoder {keras layer} -- input encoder layer
        decoder {keras layer} -- input decoder layer
    Returns:
        [keras layer] -- [output layer]
    '''
    channels = encoder.shape[-1]
    shortcut = encoder
    shortcut = BatchNormalization()(Conv2D(channels, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut))

    out = ReLU()(BatchNormalization()(Conv2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')
                                      (encoder)))

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = BatchNormalization()(Conv2D(channels, (1, 1), padding='same', kernel_initializer='he_normal')(shortcut))

        out = ReLU()(BatchNormalization()(Conv2D(channels, (3, 3), padding='same', kernel_initializer='he_normal')
                                      (out)))

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def AttUnet2d(input_size1=(224, 224, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # ConvMix layer 1
    x1 = conv_stem(input_model1, filters=256, patch_size=1)
    for _ in range(8):
        x1 = conv_mixer_block(x1, filters=256, kernel_size=9)
    # Conv Layer 1
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1)))
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


def ConvMixer2(input_size1=(160, 160, 1), n_filt=32):

    input_model1 = Input(input_size1)

    #x1 = conv_stem(input_model1, filters=256, patch_size=1)
    #for _ in range(10):
    #    x1 = conv_mixer_block(x1, filters=256, kernel_size=10)

    # layer1
    conv1 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (5, 5), padding='same', kernel_initializer='he_normal')(input_model1)))
    conv1 = conv_mixer_block(conv1, filters=n_filt, kernel_size=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # layer2
    x2 = conv_stem(input_model1, filters=256, patch_size=2)
    for _ in range(10):
        x2 = conv_mixer_block(x2, filters=256, kernel_size=10)

    conc2 = concatenate([pool1, x2], axis=3)
    conv2 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(10):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=10)

    conc3 = concatenate([pool2, x3], axis=3)
    conv3 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv3 = conv_mixer_block(conv3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # layer4
    conv4 = conv_mixer_block(pool3, filters=n_filt * 8, kernel_size=3)
    conv4 = conv_mixer_block(conv4, filters=n_filt * 8, kernel_size=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # layer5
    conv5 = conv_mixer_block(pool4, filters=n_filt * 16, kernel_size=3)
    conv5 = conv_mixer_block(conv5, filters=n_filt * 16, kernel_size=3)

    up4 = UpSampling2D(size=(2, 2))(conv5)
    skip4 = ResPath(conv4, length=1)
    conc6 = concatenate([up4, skip4], axis=3)
    conv6 = conv_mixer_block(conc6, filters=n_filt * 8, kernel_size=3)
    conv6 = conv_mixer_block(conv6, filters=n_filt * 8, kernel_size=3)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    skip3 = ResPath(conv3, length=2)
    conc7 = concatenate([up3, skip3], axis=3)
    conv7 = conv_mixer_block(conc7, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = ResPath(conv2, length=3)
    conc8 = concatenate([up2, skip2], axis=3)
    conv8 = conv_mixer_block(conc8, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = ResPath(conv1, length=4)
    conc9 = concatenate([up1, skip1], axis=3)
    conv9 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conc9)))
    conv9 = ReLU()(
        BatchNormalization()(Conv2D(n_filt, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)))

    conv_out = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=input_model1, outputs=conv_out)
    logging.info('Finish building model')
    return model

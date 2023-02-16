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


def skip_mix(x, length=1):
    channels = x.shape[-1]
    for _ in range(length):
        x = conv_mixer_block(x, filters=channels, kernel_size=3)
    return x


def ConvMixer1(input_size1=(192, 192, 1), n_filt=32):
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
    '''
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(8):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=7)
    '''
    # COnv layer3
    x32 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    x32 = conv_mixer_block(x32, filters=n_filt * 4, kernel_size=3)

    #pool3 = MaxPooling2D(pool_size=(2, 2))(concatenate([x3, x32], axis=-1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(x32)

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


def ConvMixer2(input_size1=(192, 192, 1), n_filt=32):

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
    for _ in range(8):
        x2 = conv_mixer_block(x2, filters=256, kernel_size=7)

    conc2 = concatenate([pool1, x2], axis=3)
    conv2 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv2 = conv_mixer_block(conv2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # layer3
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(8):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=7)

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


def ConvMixer3(input_size1=(192, 192, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # ConvMix layer 1
    x1 = conv_stem(input_model1, filters=n_filt, patch_size=1)
    for _ in range(6):
        x1 = conv_mixer_block(x1, filters=n_filt, kernel_size=9)
    # Conv Layer 1
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1)))
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x12)))

    pool1 = MaxPooling2D(pool_size=(2, 2))(add([x1, x12]))

    # ConvMix layer 2
    x2 = conv_stem(input_model1, filters=n_filt * 2, patch_size=2)
    for _ in range(6):
        x2 = conv_mixer_block(x2, filters=n_filt * 2, kernel_size=7)

    # Conv Layer 2
    x22 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    x22 = conv_mixer_block(x22, filters=n_filt * 2, kernel_size=3)

    pool2 = MaxPooling2D(pool_size=(2, 2))(add([x2, x22]))

    # ConvMix layer 3
    '''
    x3 = conv_stem(input_model1, filters=256, patch_size=4)
    for _ in range(8):
        x3 = conv_mixer_block(x3, filters=256, kernel_size=7)
    '''
    # COnv layer3
    x32 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    x32 = conv_mixer_block(x32, filters=n_filt * 4, kernel_size=3)

    #pool3 = MaxPooling2D(pool_size=(2, 2))(concatenate([x3, x32], axis=-1))
    pool3 = MaxPooling2D(pool_size=(2, 2))(x32)

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

def ConvMixer4(input_size1=(192, 192, 1), n_filt=32):
    input_model1 = Input(input_size1)

    # ConvMix layer 1
    mix1 = conv_stem(input_model1, filters=128, patch_size=1)
    for _ in range(8):
        mix1 = conv_mixer_block(mix1, filters=128, kernel_size=9)
    x11 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(mix1)))
    x12 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', strides=(2, 2), kernel_initializer='he_normal')(mix1)))

    # Conv Layer 1
    x1 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(input_model1)))
    x1 = BatchNormalization()(ReLU()(Conv2D(n_filt, 3, padding='same', kernel_initializer='he_normal')(x1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(concatenate([x1, x11], axis=3))

    # Conv Layer 2
    x2 = conv_mixer_block(pool1, filters=n_filt * 2, kernel_size=3)
    x2 = conv_mixer_block(x2, filters=n_filt * 2, kernel_size=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(concatenate([x2, x12], axis=3))

    # COnv layer3
    x3 = conv_mixer_block(pool2, filters=n_filt * 4, kernel_size=3)
    x3 = conv_mixer_block(x3, filters=n_filt * 4, kernel_size=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)

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
    skip3 = ResPath(x3, length=2)
    conc3 = concatenate([up3, skip3], axis=3)

    conv7 = conv_mixer_block(conc3, filters=n_filt * 4, kernel_size=3)
    conv7 = conv_mixer_block(conv7, filters=n_filt * 4, kernel_size=3)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    skip2 = ResPath(x2, length=3)
    conc2 = concatenate([up2, skip2], axis=3)

    conv8 = conv_mixer_block(conc2, filters=n_filt * 2, kernel_size=3)
    conv8 = conv_mixer_block(conv8, filters=n_filt * 2, kernel_size=3)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    skip1 = ResPath(x1, length=4)
    conc1 = concatenate([up1, skip1], axis=3)

    conv9 = conv_mixer_block(conc1, filters=n_filt, kernel_size=3)
    conv9 = conv_mixer_block(conv9, filters=n_filt, kernel_size=3)

    conv_out = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(inputs=input_model1, outputs=conv_out)
    logging.info('Finish building model')
    return model


# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	'''
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	'''
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g


# define generator model (ResNet)
def define_generator_1(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model
    
 
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
 

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator_2(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block(b, e5, 512)
    d2 = decoder_block(d1, e4, 512, dropout=False)
    d3 = decoder_block(d2, e3, 256, dropout=False)
    d4 = decoder_block(d3, e2, 128, dropout=False)
    d5 = decoder_block(d4, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

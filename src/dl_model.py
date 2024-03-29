"""
Code borrowed and modified from https://github.com/shenshutao/image_segmentation/blob/master/models/deeplabv3plus.py
"""

from keras.engine import Layer
from keras import layers, models
import tensorflow as tf
import keras.backend as K


class BilinearResizeLayer2D(Layer):
    """
    Instead of using Lambda layer, custom layer is a better practice.
    And Lambda will got Serialization problem during model save.
    """

    def __init__(self, target_size, **kwargs):
        self.target_size = target_size
        super(BilinearResizeLayer2D, self).__init__(**kwargs)

    def call(self, x):
        return tf.image.resize_bilinear(x, size=self.target_size)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.target_size[0], self.target_size[1], input_shape[3]

    def get_config(self):  # For model serialization
        config = super(BilinearResizeLayer2D, self).get_config()
        if hasattr(self, 'target_size'):
            config['target_size'] = self.target_size
        return config


def separableConv2DWithBN(filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1),
                          activation_fn_in_separable_conv=False, name=None):
    """
    The Separable Conv 2D in Deeplab V3plus is a bit different with the Keras one, it has a BN layer after the
    Depthwise Conv Layer,
    and it might have activation func between the Depthwise Layer & Pointwise Layer.
    """
    depthwise_name, pointwise_name, output_bn = (name + '_depthwise', name + '_pointwise', name + '_bn') if name else (
        None, None, None)

    def call(x):
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same',
                                   kernel_initializer='he_normal', use_bias=False, name=depthwise_name)(x)
        x = layers.BatchNormalization()(x)
        if activation_fn_in_separable_conv:
            x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,
                          name=pointwise_name)(x)
        x = layers.BatchNormalization(name=output_bn)(x)

        return x

    return call


def xception_moudle(x, prefix, depth_list, skip_connection_type='conv', activation_fn_in_separable_conv=False,
                    ds_strides=(1, 1)):
    # Prepare shortcut
    if skip_connection_type == 'conv':
        residual = layers.Conv2D(depth_list[2], (1, 1), strides=ds_strides, padding='same', use_bias=False,
                                 kernel_initializer='he_normal')(x)
        residual = layers.BatchNormalization()(residual)
    elif skip_connection_type == 'sum':
        residual = x

    x = layers.Activation('relu')(x)
    x = separableConv2DWithBN(name=prefix + '_c1', filters=depth_list[0],
                              activation_fn_in_separable_conv=activation_fn_in_separable_conv)(x)
    x = layers.Activation('relu')(x)
    x = separableConv2DWithBN(name=prefix + '_c2', filters=depth_list[1],
                              activation_fn_in_separable_conv=activation_fn_in_separable_conv)(x)
    x = layers.Activation('relu')(x)
    x = separableConv2DWithBN(name=prefix + '_c3', filters=depth_list[2], strides=ds_strides,
                              activation_fn_in_separable_conv=activation_fn_in_separable_conv)(x)
    if skip_connection_type in ('conv', 'sum'):
        x = layers.add([x, residual])
    return x


def get_enhanced_xception(input_tensor, count_mid_flow=16):
    img_input = input_tensor
    with tf.variable_scope("Xception_65"):
        with tf.variable_scope("entry_flow"):
            with tf.variable_scope("conv_1_1"):
                x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='entry_conv_1',
                                  kernel_initializer='he_normal')(
                    img_input)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

            with tf.variable_scope("conv_1_2"):
                x = layers.Conv2D(64, (3, 3), use_bias=False, padding='same', name='entry_conv_2',
                                  kernel_initializer='he_normal')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

            with tf.variable_scope("block_1"):
                x = xception_moudle(x, prefix='entry_block1', depth_list=(128, 128, 128),
                                    skip_connection_type='conv', ds_strides=(2, 2))

            with tf.variable_scope("block_2"):
                x = xception_moudle(x, prefix='entry_block2', depth_list=(256, 256, 256),
                                    skip_connection_type='conv', ds_strides=(2, 2))

            with tf.variable_scope("block_3"):
                x = xception_moudle(x, prefix='entry_block3', depth_list=(728, 728, 728),
                                    skip_connection_type='conv', ds_strides=(2, 2))

        with tf.variable_scope("middle_flow"):
            for i in range(count_mid_flow):
                prefix = 'unit_' + str(i + 1)
                with tf.variable_scope(prefix):
                    x = xception_moudle(x, prefix=prefix, depth_list=(728, 728, 728),
                                        skip_connection_type='sum', ds_strides=(1, 1))

        with tf.variable_scope("exit_flow"):
            with tf.variable_scope('block_1'):
                x = xception_moudle(x, prefix='exit_b1', depth_list=(728, 1024, 1024),
                                    skip_connection_type='conv', ds_strides=(
                        1, 1))  # In paper only stride 16, so no stride here, keep the dimension as 33x33

            with tf.variable_scope('block_2'):
                x = xception_moudle(x, prefix='exit_b2', depth_list=(1536, 1536, 2048),
                                    skip_connection_type='none', ds_strides=(1, 1))

    model = models.Model(inputs=img_input, outputs=x, name='enhanced_xception')

    return model


def get_separable_atrous_conv(x_output, atrous_rate=(6, 12, 18)):
    # 1x1 Conv
    aspp0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0',
                          kernel_initializer='he_normal')(x_output)
    aspp0 = layers.BatchNormalization(epsilon=1e-5)(aspp0)
    aspp0 = layers.Activation('relu')(aspp0)

    # 3x3 Separable Conv rate 6
    aspp1 = separableConv2DWithBN(256, (3, 3), dilation_rate=(atrous_rate[0], atrous_rate[0]),
                                  activation_fn_in_separable_conv=True, name='aspp1')(x_output)
    aspp1 = layers.Activation('relu')(aspp1)

    # 3x3 Separable Conv rate 12
    aspp2 = separableConv2DWithBN(256, (3, 3), dilation_rate=(atrous_rate[1], atrous_rate[1]),
                                  activation_fn_in_separable_conv=True, name='aspp2')(x_output)
    aspp2 = layers.Activation('relu')(aspp2)

    # 3x3 Separable Conv rate 18
    aspp3 = separableConv2DWithBN(256, (3, 3), dilation_rate=(atrous_rate[2], atrous_rate[2]),
                                  activation_fn_in_separable_conv=True, name='aspp3')(x_output)
    aspp3 = layers.Activation('relu')(aspp3)

    # Image Pooling
    aspp4 = layers.GlobalAveragePooling2D()(x_output)
    aspp4 = layers.Reshape((1, 1, -1))(aspp4)
    aspp4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling',
                          kernel_initializer='he_normal')(aspp4)  # 减少层数
    aspp4 = layers.Activation('relu')(aspp4)
    aspp4 = BilinearResizeLayer2D(target_size=(K.int_shape(x_output)[1], K.int_shape(x_output)[2]),
                                  name='UpSampling_aspp4')(aspp4)  # Reshape back for concat

    x = layers.Concatenate()([aspp0, aspp1, aspp2, aspp3, aspp4])
    return x


def get_model(input_shape=(513, 513, 3), output_dims=(513, 513), atrous_rate=(6, 12, 18), class_no=21, crop=False,
              crop_size=None, count_mid_flow=16):
    input_tensor = layers.Input(shape=input_shape)
    with tf.variable_scope("encoder"):
        encoder = get_enhanced_xception(input_tensor=input_tensor, count_mid_flow=count_mid_flow)
        x_output = encoder.output

        # for layer in encoder.layers:  #  not available as pre train model is not ready here.
        #     layer.trainable = False

        x = get_separable_atrous_conv(x_output, atrous_rate=atrous_rate)

        x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection',
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

    with tf.variable_scope("decoder"):
        # x4 (x2) block
        skip1 = encoder.get_layer('entry_block2_c2_bn').output

        x = BilinearResizeLayer2D(target_size=(K.int_shape(skip1)[1], K.int_shape(skip1)[2]), name='UpSampling1')(x)

        dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0',
                                  kernel_initializer='he_normal')(skip1)
        dec_skip1 = layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = layers.Activation('relu')(dec_skip1)
        x = layers.Concatenate()([x, dec_skip1])

        x = layers.Conv2D(class_no, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BilinearResizeLayer2D(target_size=output_dims, name='UpSampling2')(x)
        if crop:
            x = layers.Cropping2D(crop_size)(x)
    if class_no == 1:
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    model = models.Model(inputs=input_tensor, outputs=x, name='deeplab_v3plus')

    return model

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras import Input
from keras.layers import Lambda


class SRGAN(object):
    @staticmethod
    def generator(scaling_factor, feature_maps, residual_blocks):
        """
        Construct a generator model for Super-Resolution Generative Adversarial Network (SRGAN).

        Parameters:
        scaling_factor (int): Scaling factor for super-resolution.
        feature_maps (int): Number of feature maps in the generator.
        residual_blocks (int): Number of residual blocks in the generator.

        Returns:
        Model: Generator model for SRGAN.
        """
        inputs = Input((None, None, 3))
        x_in = Rescaling(scale=(1.0 / 255.0), offset=0.0)(inputs)
        x_in = Conv2D(feature_maps, 9, padding="same")(x_in)
        x_in = PReLU(shared_axes=[1, 2])(x_in)

        # construct the "residual in residual" block
        x = Conv2D(feature_maps, 3, padding="same")(x_in)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_maps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x_skip = Add()([x_in, x])

        # create a number of residual blocks
        for _ in range(residual_blocks - 1):
            x = Conv2D(feature_maps, 3, padding="same")(x_skip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_maps, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x_skip = Add()([x_skip, x])

        # get the last residual block without activation
        x = Conv2D(feature_maps, 3, padding="same")(x_skip)
        x = BatchNormalization()(x)
        x = Add()([x_in, x])

        # upscale the image with pixel shuffle
        x = Conv2D(feature_maps * (scaling_factor // 2), 3, padding="same")(x)
        depth_to_space_layer = Lambda(lambda x: tf.nn.depth_to_space(x, 2))
        x = depth_to_space_layer(x)
        x = PReLU(shared_axes=[1, 2])(x)

        # upscale the image with pixel shuffle
        x = Conv2D(feature_maps * scaling_factor, 3,
                   padding="same")(x)
        depth_to_space_layer = Lambda(lambda x: tf.nn.depth_to_space(x, 2))
        x = depth_to_space_layer(x)
        x = PReLU(shared_axes=[1, 2])(x)

        # get the output and scale it from [-1, 1] to [0, 255] range
        x = Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)

        generator = Model(inputs, x)
        return generator

    @staticmethod
    def discriminator(feature_maps, leaky_alpha, disc_blocks):
        """
        Construct a discriminator model for Super-Resolution Generative Adversarial Network (SRGAN).

        Parameters:
        feature_maps (int): Number of feature maps in the discriminator.
        leaky_alpha (float): Slope of the leak for LeakyReLU.
        disc_blocks (int): Number of discriminator blocks.

        Returns:
        Model: Discriminator model for SRGAN.
        """
        inputs = Input((None, None, 3))
        x = Rescaling(scale=(1.0 / 127.5), offset=-1.0)(inputs)
        x = Conv2D(feature_maps, 3, padding="same")(x)
        # unlike the generator we use leaky relu in the discriminator
        x = LeakyReLU(leaky_alpha)(x)
        x = Conv2D(feature_maps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leaky_alpha)(x)

        # create a number of discriminator blocks
        for i in range(1, disc_blocks):
            x = Conv2D(feature_maps * (2 ** i), 3, strides=2, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leaky_alpha)(x)
            x = Conv2D(feature_maps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leaky_alpha)(x)

        # process the feature maps with global average pooling
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leaky_alpha)(x)
        x = Dense(1, activation="sigmoid")(x)

        discriminator = Model(inputs, x)
        return discriminator

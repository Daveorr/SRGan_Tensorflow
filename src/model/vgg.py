from keras.applications.vgg19 import VGG19
from tensorflow.keras import Model


class VGG:
    @staticmethod
    def build():
        """
        Construct a VGG19 model for perceptual loss in Super-Resolution Generative Adversarial Network (SRGAN).
        Returns:
        Model: VGG19 model with custom output layer (VGG19 model sliced at level 20).
        """
        vgg = VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        model = Model(vgg.input, vgg.layers[20].output)
        return model

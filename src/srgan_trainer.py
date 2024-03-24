from tensorflow.keras import Model
from tensorflow import GradientTape
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
import tensorflow as tf


class SRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batch_size):
        """
        Initialize the SRGANTrainer.
        Parameters:
        generator (Model): The generator model.
        discriminator (Model): The discriminator model.
        vgg (Model): The VGG model for perceptual loss.
        batch_size (int): Batch size for training.
        """
        super().__init__()
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.mse_loss = None
        self.bce_loss = None
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batch_size = batch_size

    def compile(self, g_optimizer, d_optimizer, bce_loss, mse_loss):
        """
        Compile the SRGANTrainer with optimizer and loss functions.
        Parameters:
        g_optimizer: The optimizer for the generator.
        d_optimizer: The optimizer for the discriminator.
        bce_loss: Binary Cross-entropy loss function.
        mse_loss: Mean Squared Error loss function.
        """
        super().compile()
        self.generator_optimizer = g_optimizer
        self.discriminator_optimizer = d_optimizer
        self.bce_loss = bce_loss
        self.mse_loss = mse_loss

    def _perc_loss(self, hr_images, sr_images):
        """
        Compute the perceptual loss between high-resolution and super-resolution images using the VGG model.

        Perceptual loss measures the difference in content and style between the high-resolution (HR) and
        super-resolution (SR) images.
        It is computed by comparing the feature representations of the images extracted from a pre-trained VGG network.
        This approach encourages the generator to produce SR images that not only have high pixel-level similarity to
        HR images  but also capture the underlying structure and characteristics of the content in HR images.

        Parameters:
        hr_images (tensor): High-resolution images.
        sr_images (tensor): Super-resolution images.

        Returns:
        tensor: Perceptual loss.
        """
        sr_vgg = tf.keras.applications.vgg19.preprocess_input(sr_images)
        sr_vgg = self.vgg(sr_vgg) / 12.75
        hr_vgg = tf.keras.applications.vgg19.preprocess_input(hr_images)
        hr_vgg = self.vgg(hr_vgg) / 12.75
        return self.mse_loss(hr_vgg, sr_vgg)

    def train_step(self, images):
        """
        Perform a single training step of the SRGAN model.

        - Optimize the generator and discriminator networks iteratively.
        - Generator produces high-resolution images from low-resolution inputs, minimizing adversarial and perceptual losses.
        - Discriminator learns to distinguish between real and generated images, optimizing its classification ability.
        - Adversarial training fosters a competitive dynamic between the networks, facilitating mutual improvement.
        - Different loss functions (binary cross-entropy, mean squared error) compute adversarial and perceptual losses.
        - Gradient-based optimization algorithms adjust network parameters iteratively to minimize loss functions.

        Parameters:
        images (tuple): A tuple containing low-resolution (LR) and high-resolution (HR) images.

        Returns:
        dict: Dictionary containing discriminator and generator losses.
        """

        (lr_images, hr_images) = images
        lr_images = tf.cast(lr_images, tf.float32)
        hr_images = tf.cast(hr_images, tf.float32)
        sr_images = self.generator(lr_images)

        # label 0 is for predicted images and 1 is for original high resolution images
        combined_images = concat([sr_images, hr_images], axis=0)
        labels = concat([zeros((self.batch_size, 1)), ones((self.batch_size, 1))], axis=0)

        # train the discriminator
        with GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.bce_loss(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        # generate misleading labels
        misleading_labels = ones((self.batch_size, 1))

        # train the generator (note that we should *not* update the  weights of the discriminator)!
        with GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_images)
            predictions = self.discriminator(fake_images)
            # compute the adversarial loss
            g_loss = 1e-3 * self.bce_loss(misleading_labels, predictions)
            perc_loss = self._perc_loss(hr_images, fake_images)
            # calculate the total generator loss
            g_total_loss = g_loss + perc_loss

        grads = tape.gradient(g_total_loss, self.generator.trainable_variables)
        # optimize the generator weights with the computed gradients
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        # return the generator and discriminator losses
        return {"d_loss": d_loss, "g_total_loss": g_total_loss, "g_loss": g_loss, "perc_loss": perc_loss}

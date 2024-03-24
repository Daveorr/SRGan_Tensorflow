from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import Reduction
from tensorflow import reduce_mean


class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas

    def bce_loss(self, real, pred):
        """
        Computes the Binary Cross-Entropy loss between real and predicted values.
        Args:
            real (tf.Tensor): The real values.
            pred (tf.Tensor): The predicted values.
        Returns:
            tf.Tensor: The computed loss value.
        """
        BCE = BinaryCrossentropy(reduction=Reduction.NONE)
        loss = BCE(real, pred)
        loss = reduce_mean(loss) * (1. / self.numReplicas)
        return loss

    def mse_loss(self, real, pred):
        """
        Computes the Mean Squared Error loss between real and predicted values.
        Args:
            real (tf.Tensor): The real values.
            pred (tf.Tensor): The predicted values.
        Returns:
            tf.Tensor: The computed loss value.
        """
        MSE = MeanSquaredError(reduction=Reduction.NONE)
        loss = MSE(real, pred)
        loss = reduce_mean(loss) * (1. / self.numReplicas)
        return loss

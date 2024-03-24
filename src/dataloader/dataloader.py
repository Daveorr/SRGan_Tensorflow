import tensorflow as tf
from tensorflow.io import FixedLenFeature
from .augmentations import random_crop, random_flip, random_rotate, get_center_crop

# Define constants
AUTO = tf.data.AUTOTUNE
HR_SIZE = 96  # Original high-resolution image size
SCALE = 4  # Scaling factor


def _preprocess_image(lr_image, hr_image):
    """
    Preprocesses the given low-resolution (lr_image) and high-resolution (hr_image) TensorFlow tensors.
    Applies data augmentation functions including random cropping, flipping, and rotation.
    Args:
        lr_image (tf.Tensor): Low-resolution input image tensor.
        hr_image (tf.Tensor): High-resolution target image tensor.
    Returns:
        tuple: A tuple containing preprocessed low-resolution and high-resolution image tensors.
    """
    lr_image, hr_image = random_crop(lr_image, hr_image, HR_SIZE, SCALE)
    lr_image, hr_image = random_flip(lr_image, hr_image)
    lr_image, hr_image = random_rotate(lr_image, hr_image)
    return lr_image, hr_image


def _read_train_sample(sample, lr_img_shape, hr_img_shape):
    """
    Reads and preprocesses a training sample containing low-resolution (lr) and high-resolution (hr) images.
    Args:
        sample (tf.Tensor): A sample from the training dataset.
        lr_img_shape (tuple): Shape of the low-resolution image.
        hr_img_shape (tuple): Shape of the high-resolution image.
    Returns:
        tuple: A tuple containing preprocessed low-resolution and high-resolution image tensors.
    """
    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(sample, feature)

    lr_image = tf.image.decode_jpeg(example["lr"], channels=3)
    hr_image = tf.image.decode_jpeg(example["hr"], channels=3)
    lr_image = tf.cast(lr_image, tf.uint8)
    hr_image = tf.cast(hr_image, tf.uint8)

    lr_image_aug, hr_image_aug = _preprocess_image(lr_image, hr_image)
    lr_tensor_in = tf.reshape(lr_image_aug, lr_img_shape)
    hr_tensor_in = tf.reshape(hr_image_aug, hr_img_shape)
    return lr_tensor_in, hr_tensor_in


def _read_test_sample(sample, lr_img_shape, hr_img_shape):
    """
    Reads and preprocesses a test sample containing low-resolution (lr) and high-resolution (hr) images.
    Args:
        sample (tf.Tensor): A sample from the test dataset.
        lr_img_shape (tuple): Shape of the low-resolution image.
        hr_img_shape (tuple): Shape of the high-resolution image.
    Returns:
        tuple: A tuple containing preprocessed low-resolution and high-resolution image tensors.
    """
    feature = {
        "lr": tf.io.FixedLenFeature([], tf.string),
        "hr": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(sample, feature)

    lr_image = tf.image.decode_jpeg(example["lr"], channels=3)
    hr_image = tf.image.decode_jpeg(example["hr"], channels=3)
    lr_image = tf.cast(lr_image, tf.uint8)
    hr_image = tf.cast(hr_image, tf.uint8)

    lr_image_aug, hr_image_aug = get_center_crop(lr_image, hr_image)

    lr_tensor_in = tf.image.resize(lr_image_aug, lr_img_shape[:2])
    hr_tensor_in = tf.image.resize(hr_image_aug, hr_img_shape[:2])

    return lr_tensor_in, hr_tensor_in


def create_data_loader(tf_records_paths,
                       lr_img_shape=(24, 24, 3),
                       hr_img_shape=(96, 96, 3),
                       batch_size=32,
                       train_mode=False):
    """
    Creates a TensorFlow data loader for training or testing with TFRecord files.
    Args:
        tf_records_paths (list): List of paths to TFRecord files.
        lr_img_shape (tuple): Shape of the low-resolution images.
        hr_img_shape (tuple): Shape of the high-resolution images.
        batch_size (int): Batch size for the data loader.
        train_mode (bool): If True, creates a data loader for training; otherwise, for testing.
    Returns:
        tf.data.Dataset: A TensorFlow dataset object containing preprocessed image batches.
    """
    dataset = tf.data.TFRecordDataset(tf_records_paths, num_parallel_reads=AUTO)
    if train_mode:
        # read the training examples
        dataset = dataset.map(lambda x: _read_train_sample(x, lr_img_shape, hr_img_shape),
                              num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(lambda x: _read_test_sample(x, lr_img_shape, hr_img_shape),
                              num_parallel_calls=AUTO)
    # batch and prefetch
    dataset = (dataset
               .shuffle(batch_size)
               .batch(batch_size)
               .repeat()
               .prefetch(AUTO))
    return dataset

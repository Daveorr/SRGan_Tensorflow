import tensorflow as tf


def random_crop(lr_image, hr_image, hr_crop_size=96, scale=4):
    """
    Randomly crops the low and high resolution images.

    Args:
        lr_image (tf.Tensor): The low resolution image tensor.
        hr_image (tf.Tensor): The high resolution image tensor.
        hr_crop_size (int): The size of the cropped high resolution image. Default is 96.
        scale (int): The scaling factor between low and high resolution images. Default is 4.

    Returns:
        tuple: A tuple containing the cropped low resolution image tensor
               and the cropped high resolution image tensor.
    """
    # Calculate the low resolution image crop size and image shape
    lr_crop_size = hr_crop_size // scale
    lr_image_shape = tf.shape(lr_image)[:2]
    # Calculate the low resolution image width and height offsets
    lr_w = tf.random.uniform(shape=(),
                             maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(),
                             maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    # Calculate the high resolution image width and height
    hr_w = lr_w * scale
    hr_h = lr_h * scale
    # Crop the low and high resolution images
    lr_image_cropped = tf.slice(lr_image, [lr_h, lr_w, 0], [lr_crop_size, lr_crop_size, 3])
    hr_image_cropped = tf.slice(hr_image, [hr_h, hr_w, 0], [hr_crop_size, hr_crop_size, 3])
    # Return the cropped low and high resolution images
    return lr_image_cropped, hr_image_cropped


def get_center_crop(lr_image, hr_image, hr_crop_size=96, scale=4):
    """
    Retrieves the center crop of low and high resolution images.

    Args:
        lr_image (tf.Tensor): The low resolution image tensor.
        hr_image (tf.Tensor): The high resolution image tensor.
        hr_crop_size (int): The size of the cropped high resolution image. Default is 96.
        scale (int): The scaling factor between low and high resolution images. Default is 4.

    Returns:
        tuple: A tuple containing the cropped low resolution image tensor
               and the cropped high resolution image tensor.
    """
    # Calculate the low resolution image crop size and image shape
    lr_crop_size = hr_crop_size // scale
    lr_image_shape = tf.shape(lr_image)[:2]
    # Calculate the low resolution image width and height
    lr_w = lr_image_shape[1] // 2
    lr_h = lr_image_shape[0] // 2
    # Calculate the high resolution image width and height
    hr_w = lr_w * scale
    hr_h = lr_h * scale
    # Crop the low and high resolution images
    lr_image_cropped = tf.slice(lr_image, [lr_h - (lr_crop_size // 2),
                                           lr_w - (lr_crop_size // 2), 0],
                                [lr_crop_size, lr_crop_size, 3])
    hr_image_cropped = tf.slice(hr_image, [hr_h - (hr_crop_size // 2),
                                           hr_w - (hr_crop_size // 2), 0],
                                [hr_crop_size, hr_crop_size, 3])
    # Return the cropped low and high resolution images
    return lr_image_cropped, hr_image_cropped


def random_flip(lr_image, hr_image):
    """
    Randomly flips the low and high resolution images horizontally.

    Args:
        lr_image (tf.Tensor): The low resolution image tensor.
        hr_image (tf.Tensor): The high resolution image tensor.

    Returns:
        tuple: A tuple containing the randomly flipped low resolution image tensor
               and the randomly flipped high resolution image tensor.
    """
    # Calculate a random chance for flip
    flip_prob = tf.random.uniform(shape=(), maxval=1)
    lr_image, hr_image = tf.cond(flip_prob < 0.5,
                                 lambda: (lr_image, hr_image),
                                 lambda: (tf.image.flip_left_right(lr_image),
                                          tf.image.flip_left_right(hr_image)))
    # Return the randomly flipped low and high resolution images
    return lr_image, hr_image


def random_rotate(lr_image, hr_image):
    """
    Randomly rotates the low and high resolution images by a multiple of 90 degrees.

    Args:
        lr_image (tf.Tensor): The low resolution image tensor.
        hr_image (tf.Tensor): The high resolution image tensor.

    Returns:
        tuple: A tuple containing the randomly rotated low resolution image tensor
               and the randomly rotated high resolution image tensor.
    """
    # Randomly generate the number of 90 degree rotations
    n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Rotate the low and high resolution images
    lr_image = tf.image.rot90(lr_image, n)
    hr_image = tf.image.rot90(hr_image, n)
    # Return the randomly rotated images
    return lr_image, hr_image

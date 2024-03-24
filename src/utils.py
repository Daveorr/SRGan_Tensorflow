import tensorflow as tf


def check_GPU_is_available(is_required=False):
    if tf.test.is_built_with_cuda():
        print("Tensorflow was built with GPU support.")
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num of visible GPU(s):", len(physical_devices))
        print("Device(s) name(s):", physical_devices)
        if len(physical_devices) == 0 and is_required:
            raise Exception("GPU is required to run this code. Exiting...")
    else:
        print("Tensorflow was built without GPU support.")
        print("If this is not expected please check your env variables PATH and LD_LIBRARY_PATH")
        if is_required:
            raise Exception("GPU is required to run this code. Exiting...")


def get_tensor_from_path(image_path):
    """
    Reads an image from the given path and returns a tensor.
    Args:
        image_path (str): Path to the image file.
    Returns:
        tf.Tensor: Tensor containing the image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.uint8)
    image = tf.expand_dims(image, axis=0)
    return image


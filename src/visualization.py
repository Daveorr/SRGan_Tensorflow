import matplotlib.pyplot as plt
import tensorflow as tf


def plot_images_from_tensor(image_tensor, num_rows=2, num_cols=2, figsize=(10, 10)):
    """
    Plots a mosaic of images from a TensorFlow tensor.
    Args:
        image_tensor (tf.Tensor): TensorFlow tensor containing batch of images.
        num_rows (int): Number of rows in the mosaic.
        num_cols (int): Number of columns in the mosaic.
        figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
    """
    images = image_tensor.numpy()
    batch_size, height, width, channels = images.shape

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < batch_size:
                axes[i, j].imshow(images[idx] / 255.0)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')


def plot_imgs_in_row(image_title_list, figsize=None):
    """
    Plot images in a row with their corresponding titles.

    Parameters:
    - image_title_list (list of tuples): A list of tuples where each tuple contains an image and its title.
    - figsize (tuple, optional): Figure size in inches (width, height).

    Example:
    plot_imgs_in_row([(img1, 'Title 1'), (img2, 'Title 2')], figsize=(10, 5))
    """
    num_images = len(image_title_list)

    if figsize is None:
        fig, axes = plt.subplots(1, num_images)
    else:
        fig, axes = plt.subplots(1, num_images, figsize=figsize)

    for i, (img, title) in enumerate(image_title_list):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

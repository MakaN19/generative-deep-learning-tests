import tensorflow as tf
from tensorflow.keras import callbacks

import numpy as np
import matplotlib.pyplot as plt


def display(
    images: np.array, n: int = 10, size: tuple[int] = (20, 3),
    cmap="gray_r", as_type: str = "float32", save_to=None) -> None:

    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img: int, latent_dim: int) -> None:
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim)
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = generated_images * 127.5 + 127.5
        generated_images = generated_images.numpy()
        display(
            generated_images,
            save_to="./output/generated_img_%03d.png" % (epoch),)

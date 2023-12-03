import math

import matplotlib.pyplot as plt
import numpy as np


def plot_embeddings2D(encoder_predictions: np.array) -> None:
    plt.figure(figsize=(8, 8))

    plt.scatter(encoder_predictions[:, 0], encoder_predictions[:, 1], c="black", alpha=0.4, s=2)
    plt.show()

def plot_reconstructions(encoder_predictions, reconstructions: np.array) -> None:
    batch_size, _ = encoder_predictions.shape

    width = 6
    height = math.ceil(batch_size / width)

    fig = plt.figure(figsize=(8, height * 2))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(batch_size):
        ax = fig.add_subplot(height, width, i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            str(np.round(encoder_predictions[i, :], 1)),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(reconstructions[i, :, :], cmap="Greys")

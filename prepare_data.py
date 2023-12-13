from tensorflow.keras import datasets
import numpy as np


def prepare_fashion_mnist_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    x_train = np.expand_dims(x_train, -1)

    x_test = x_test.astype("float32") / 255.0
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test

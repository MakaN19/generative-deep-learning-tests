from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K

import numpy as np

def build_encoder(
        image_height: int,
        image_width: int,
        n_channels: int,
        embedding_dim: int) -> tuple[models.Model, tuple[int], layers.Input, layers.Input]:

    encoder_in = layers.Input(
        shape=(image_height, image_width, n_channels), name="encoder_in"
    )
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_in)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_before_flatten = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    encoder_out = layers.Dense(embedding_dim, name="encoder_out")(x)

    encoder = models.Model(encoder_in, encoder_out)
    print(encoder.summary())

    return encoder, shape_before_flatten, encoder_in, encoder_out


def build_decoder(
        n_channels: int,
        embedding_dim: int,
        shape_before_flatten: tuple[int]) -> models.Model:

    decoder_in = layers.Input(shape=(embedding_dim,), name="decoder_in")
    x = layers.Dense(np.prod(shape_before_flatten))(decoder_in)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoder_out = layers.Conv2D(
        n_channels,
        (3, 3),
        strides=1,
        activation="sigmoid",
        padding="same",
        name="decoder_out",
    )(x)

    decoder = models.Model(decoder_in, decoder_out)
    print(decoder.summary())

    return decoder


def build_autoencoder(
        image_height: int,
        image_width: int,
        n_channels: int,
        embedding_dim: int) -> models.Model:

    encoder, shape_before_flatten, encoder_in, encoder_out = build_encoder(
        image_height, image_width, n_channels, embedding_dim)
    decoder = build_decoder(n_channels, embedding_dim, shape_before_flatten)

    encoder_decoder = models.Model(
        encoder_in, decoder(encoder_out)
    )
    print(encoder_decoder.summary())

    return encoder_decoder, encoder, decoder

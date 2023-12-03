from tensorflow.keras import layers, models, metrics, losses
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np


class SamplingLayer(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(models.Model):
    def __init__(self, encoder: models.Model, decoder: models.Model, beta=500, **kwargs) -> None:
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss = metrics.Mean(name="total_loss")
        self.reconstruction_loss = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss = metrics.Mean(name="kl_loss")
        self.beta = beta


    def call(self, inputs: tf.Tensor) -> None:
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data: tf.Tensor) -> dict:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                self.beta
                * losses.binary_crossentropy(
                    data, reconstruction, axis=(1, 2, 3)
                )
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss.update_state(total_loss)
        self.reconstruction_loss.update_state(reconstruction_loss)
        self.kl_loss.update_state(kl_loss)

        return {m.name: m.result() for m in (self.total_loss, self.reconstruction_loss, self.kl_loss)}

    def test_step(self, data: tf.Tensor) -> dict:
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            self.beta
            * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def build_encoder(
        image_height: int,
        image_width: int,
        n_channels: int,
        embedding_dim: int) -> tuple[models.Model, tuple[int]]:

    encoder_in = layers.Input(
        shape=(image_height, image_width, n_channels), name="encoder_in"
    )
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_in)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_before_flatten = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    encoder_out_mean = layers.Dense(embedding_dim, name="encoder_out_mean")(x)
    encoder_out_log_var = layers.Dense(embedding_dim, name="encoder_out_log_var")(x)
    encoder_out_sample = SamplingLayer()([encoder_out_mean, encoder_out_log_var])

    encoder = models.Model(encoder_in, [encoder_out_mean, encoder_out_log_var, encoder_out_sample])
    print(encoder.summary())

    return encoder, shape_before_flatten


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

    encoder, shape_before_flatten = build_encoder(
        image_height, image_width, n_channels, embedding_dim)
    decoder = build_decoder(n_channels, embedding_dim, shape_before_flatten)

    encoder_decoder = VariationalAutoEncoder(encoder, decoder)

    return encoder_decoder, encoder, decoder

from tensorflow.keras import models, layers, metrics, optimizers
import tensorflow as tf


def build_generator(z_dim: int) -> models.Model:
    inp = layers.Input(shape=(1, 1, z_dim))
    x = tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu")(inp)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    )(x)
    generator = models.Model(inputs=inp, outputs=x)

    return generator

def build_discriminator() -> models.Model:
    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    )(inp)
    x =tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    discriminator = models.Model(inputs=inp, outputs=x)

    return discriminator


class WassersteinGANGradientPenalty(models.Model):
    def __init__(self, critic: models.Model, generator: models.Model, latent_dim: int, gp_weight: float) -> None:
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, c_optimizer: optimizers.Optimizer, g_optimizer: optimizers.Optimizer):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_metric = metrics.Mean(name="c_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        x_hat = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat, training=True)

        gradients = t.gradient(d_hat, x_hat)

        # 1.0e-12 ! unstable othwerwise!
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]) + 1.0e-12)
        gp = tf.reduce_mean((ddx - 1.0) ** 2)

        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        z_samp = tf.random.normal([batch_size, 1, 1, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images = self.generator(z_samp, training=True)

            logits_real = self.critic(real_images, training=True)
            logits_gen = self.critic(gen_images, training=True)

            d_regularizer = self.gradient_penalty(batch_size, real_images, gen_images)
            disc_loss = (
                    tf.reduce_mean(logits_real)
                    - tf.reduce_mean(logits_gen)
                    + d_regularizer * self.gp_weight
            )

            gen_loss = tf.reduce_mean(logits_gen)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        disc_gradients = disc_tape.gradient(disc_loss, self.critic.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.c_optimizer.apply_gradients(
            zip(disc_gradients, self.critic.trainable_variables)
        )

        self.c_loss_metric.update_state(disc_loss)
        self.g_loss_metric.update_state(gen_loss)

        return {m.name: m.result() for m in self.metrics}

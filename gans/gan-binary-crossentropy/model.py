from tensorflow.keras import layers, models, losses, metrics
import tensorflow as tf


def build_generator(print_summary: bool = False) -> models.Model:

    generator_input = layers.Input(shape=(None, 100))
    x = layers.Reshape((1, 1, 100))(generator_input)

    x = layers.Conv2DTranspose(128, strides=2, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(64, strides=2, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(32, strides=2, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(16, strides=2, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(1, strides=2, kernel_size=3,
                               padding="SAME", activation="sigmoid")(x)

    generator = models.Model(inputs=generator_input, outputs=x)

    if print_summary:
        print(generator.summary())

    return generator


def build_discriminator(print_summary: bool = False) -> models.Model:

    discriminator_input = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(16, strides=3, kernel_size=3, padding="SAME")(discriminator_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, strides=3, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, strides=3, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, strides=3, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, strides=3, kernel_size=3, padding="SAME")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)


    discriminator = models.Model(inputs=discriminator_input, outputs=x)

    if print_summary:
        print(discriminator.summary())

    return discriminator


class GenerativeAdversarialNetwork(models.Model):
    def __init__(self, discriminator: models.Model,
                 generator: models.Model, gen_input_dim: int):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_input_dim = gen_input_dim

        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_fn = None
        self.d_loss_metric = None
        self.d_accuracy = None


    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = losses.BinaryCrossentropy()

        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.d_accuracy = metrics.BinaryAccuracy(name="d_accuracy")


    @property
    def metrics(self):
        return [
            self.d_loss_metric
        ]

    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        gen_input = tf.random.normal(
            shape=(batch_size, self.gen_input_dim)
        )
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            # get predictions for real images
            real_imgs_preds = self.discriminator(
                real_imgs, training=True)
            real_imgs_labels = tf.ones_like(real_imgs_preds)

            # get predictions from images from generator
            gen_imgs = self.generator(
                gen_input, training=True)
            gen_imgs_preds = self.discriminator(
                gen_imgs, training=True)

            gen_imgs_labels = tf.zeros_like(gen_imgs_preds)

            d_real_imgs_loss = self.loss_fn(real_imgs_labels + 0.15 * tf.random.uniform(
                tf.shape(real_imgs_preds)), real_imgs_preds)

            d_gen_imgs_loss = self.loss_fn(gen_imgs_labels + 0.15 * tf.random.uniform(
                tf.shape(gen_imgs_preds)), gen_imgs_preds)

            d_loss = (d_real_imgs_loss + d_gen_imgs_loss) / 2.0

            g_loss = self.loss_fn(real_imgs_labels, gen_imgs_preds)

        d_grad = d_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        g_grad = g_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.d_accuracy.update_state(
            [real_imgs_labels, gen_imgs_labels], [real_imgs_preds, gen_imgs_preds]
        )

        return {m.name: m.result() for m in self.metrics}

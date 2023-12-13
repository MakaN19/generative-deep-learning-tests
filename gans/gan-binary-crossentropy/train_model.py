from tensorflow.keras import datasets, callbacks, optimizers

from model import GenerativeAdversarialNetwork, build_discriminator, build_generator
from custom_callbacks import ImageGenerator
from prepare_data import prepare_fashion_mnist_data

import logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
)


IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 300
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002

logging.info("Preparing the data...")
x_train, x_test = prepare_fashion_mnist_data()

logging.info("Build generator...")
generator = build_generator(print_summary=True)

logging.info("Build discriminator...")
discriminator = build_discriminator(print_summary=True)

logging.info("Build GAN...")
gan = GenerativeAdversarialNetwork(
    discriminator=discriminator, generator=generator, gen_input_dim=100
)
gan.compile(
    d_optimizer=optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2
    ),
    g_optimizer=optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2
    ),
)


model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

logging.info("Fit GAN...")
gan.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        ImageGenerator(num_img=10, latent_dim=100),
    ],
)

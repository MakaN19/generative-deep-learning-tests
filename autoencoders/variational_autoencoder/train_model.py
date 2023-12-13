from tensorflow.keras import datasets, callbacks, optimizers

from model import build_decoder, build_encoder, VariationalAutoEncoder
from prepare_data import prepare_fashion_mnist_data

import logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
)


IMAGE_SIZE = 32
N_CHANNELS = 1
EMBEDDING_DIM = 2

logging.info("Prepare data...")
x_train, x_test = prepare_fashion_mnist_data()


logging.info("Build encoder...")
encoder, shape_before_flatten = build_encoder(
        IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS, EMBEDDING_DIM)


logging.info("Build decoder...")
decoder = build_decoder(N_CHANNELS, EMBEDDING_DIM, shape_before_flatten)


logging.info("Build Variational Auto Encoder...")

model = VariationalAutoEncoder(encoder, decoder)
optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    save_best_only=True,
    verbose=0
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./tensorboard-logs")


logging.info("Fit Variational Auto Encoder...")

model.fit(
    x_train,
    epochs=3,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

model.save("./models/autoencoder")
encoder.save("./models/encoder")
decoder.save("./models/decoder")

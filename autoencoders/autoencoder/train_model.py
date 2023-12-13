from tensorflow.keras import callbacks, models

from model import build_encoder, build_decoder
from prepare_data import prepare_fashion_mnist_data

import logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
)


IMAGE_SIZE = 32
N_CHANNELS = 3
EMBEDDING_DIM = 2

logging.info("Prepare data...")
x_train, x_test = prepare_fashion_mnist_data()


logging.info("Build encoder...")
encoder, shape_before_flatten, encoder_in, encoder_out = build_encoder(
        IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS, EMBEDDING_DIM, print_summary=True)


logging.info("Build decoder...")
decoder = build_decoder(N_CHANNELS, EMBEDDING_DIM, shape_before_flatten, print_summary=True)


logging.info("Build model...")
model = models.Model(
        encoder_in, decoder(encoder_out)
    )
model.compile(optimizer="adam", loss="binary_crossentropy")

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    save_best_only=True,
    verbose=0
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./tensorboard-logs")

logging.info("Fit model...")
model.fit(
    x_train,
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

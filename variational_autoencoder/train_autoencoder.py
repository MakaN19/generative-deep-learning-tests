from tensorflow.keras import datasets, callbacks, optimizers
import numpy as np

from vae import build_autoencoder

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
x_train = np.expand_dims(x_train, -1)

x_test = x_test.astype("float32") / 255.0
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
x_test = np.expand_dims(x_test, -1)

autoencoder, encoder, decoder = build_autoencoder(*x_train.shape[1:], embedding_dim=2)
optimizer = optimizers.Adam(learning_rate=0.0005)
autoencoder.compile(optimizer=optimizer)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    save_best_only=True,
    verbose=0
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./tensorboard-logs")

autoencoder.fit(
    x_train,
    epochs=3,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[model_checkpoint_callback, tensorboard_callback],
)

autoencoder.save("./models/autoencoder")
encoder.save("./models/encoder")
decoder.save("./models/decoder")

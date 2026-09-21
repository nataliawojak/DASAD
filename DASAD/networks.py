import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, LayerNormalization, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam


def build_encoder(input_shape, config):
    latent_dim = int(config["latent_dim"])
    learning_rate = float(config["learning_rate"])

    model = Sequential(name="encoder")
    model.add(Input(shape=input_shape))

    model.add(Dense(64))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(32))
    model.add(LayerNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Linear latent features avoid the saturated sigmoid bottleneck.
    model.add(Dense(latent_dim, activation=None, name="latent"))
    model.add(LayerNormalization())

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def build_discriminator(config):
    latent_dim = int(config["latent_dim"])
    discriminator_learning_rate = float(
        config["discriminator_learning_rate"]
    )

    model = Sequential(name="discriminator")
    model.add(Input(shape=(latent_dim,)))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation="sigmoid", name="domain_output"))

    model.compile(
        optimizer=Adam(learning_rate=discriminator_learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="domain_accuracy")],
    )
    return model


def build_task(config):
    latent_dim = int(config["latent_dim"])
    learning_rate = float(config["learning_rate"])
    gamma = float(config["gamma"])
    alpha = float(config["alpha"])

    model = Sequential(name="task")
    model.add(Input(shape=(latent_dim,)))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation="sigmoid", name="anomaly_probability"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryFocalCrossentropy(
            gamma=gamma,
            alpha=alpha,
            apply_class_balancing=True,
            from_logits=False,
        ),
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model

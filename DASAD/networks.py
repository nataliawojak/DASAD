from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from focal_loss import BinaryFocalLoss


def build_encoder(input_shape, config):
    
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=input_shape))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=BinaryFocalLoss(gamma=config["gamma"])
    )
    return model


def build_task(config):

    model = Sequential()
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=BinaryFocalLoss(gamma=config["gamma"])
    )

    return model


def build_discriminator(config):

    model = Sequential()
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=BinaryFocalLoss(gamma=config["gamma"])
    )

    return model
from tensorflow import keras

from utils.common import get_config


def get_model():
    img_size = [get_config("model.height"), get_config("model.width"), get_config("model.channel")]

    img1 = x1 = keras.Input(img_size)
    img2 = x2 = keras.Input(img_size)
    for _filter in get_config("model.conv_filters"):
        x1 = keras.layers.Conv2D(_filter, get_config("model.kernel"), padding="same")(x1)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.Dropout(get_config("model.dropout_rate"))(x1)
        x2 = keras.layers.Conv2D(_filter, get_config("model.kernel"), padding="same")(x2)
        x2 = keras.layers.BatchNormalization()(x2)
        x2 = keras.layers.Dropout(get_config("model.dropout_rate"))(x2)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Dense(get_config("model.dense_out"))(x1)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(get_config("model.dense_out"))(x2)
    y = x2 - x1
    y = keras.layers.Dense(1, activation="tanh")(y)

    return keras.Model(inputs=[img1, img2], outputs=y)

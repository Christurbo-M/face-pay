import tensorflow as tf


def data_rescale(x):
    return tf.subtract(tf.divide(x, 127.5), 1)


def inverse_rescale(y):
    return tf.round(tf.multiply(tf.add(y, 1), 127.5))

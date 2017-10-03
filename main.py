import tensorflow as tf
from tensorflow import keras

from model import get_model
from utils.common import setup_graphviz

print(tf.__version__, keras.__version__)
setup_graphviz()
keras.utils.plot_model(get_model(), "my_model.png")

import logging
import os
from glob import glob
from os import path

import tensorflow as tf

from utils.common import get_config
from utils.math import data_rescale


class DatasetCFP:
    people_list = []
    img_list = []
    dataset = None

    def __init__(self):
        logging.info(" - Initializing Dataset...")
        self.base_dir = get_config("dataset.cfp")
        self.load_filename()
        self.make_dataset()

    def load_filename(self):
        dir_list = os.listdir(self.base_dir)
        for _dir in dir_list:
            for img in glob(path.join(self.base_dir, _dir, "**", "*.jpg")):
                self.people_list.append(int(_dir))
                self.img_list.append(img)

    def make_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.people_list, self.img_list))
        dataset = dataset.map(map_func=self._parse_img, num_parallel_calls=get_config("threads"))
        dataset = dataset.shuffle(buffer_size=get_config("train.shuffle"))
        dataset = dataset.batch(batch_size=get_config("train.batch_size"))
        self.dataset = dataset.prefetch(buffer_size=get_config("train.prefetch"))

    @staticmethod
    def _parse_img(people, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, get_config("model.channel"))

        image.set_shape([get_config("model.height"), get_config("model.width"), get_config("model.channel")])
        image = tf.cast(image, tf.float32)
        image = data_rescale(image)
        return people, image

    def get_new_iterator(self):
        return tf.compat.v1.data.make_one_shot_iterator(self.dataset)

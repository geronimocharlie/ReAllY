import numpy as np
import random
import os
import tensorflow as tf


class Replay_buffer:
    def __init__(self, size, keys):
        self.buffer = {}
        for k in keys:
            self.buffer[k] = []
        self.size = size
        self.keys = keys

    def put(self, data_dict):
        dict_keys = list(data_dict.keys())

        current_len = len(self.buffer[self.keys[0]])
        add_len = len(data_dict[dict_keys[0]])
        new_len = current_len + add_len

        if new_len >= self.size:
            pop_len = new_len - self.size

            for k in self.buffer.keys():
                self.buffer[k] = self.buffer[k][pop_len:]

        for k in dict_keys:
            self.buffer[k].extend(data_dict[k])

        return self.buffer

    def sample(self, num):

        seed = random.randint(0, 100)
        sample = {}
        for k in self.buffer.keys():
            random.seed(seed)
            sample[k] = np.asarray(random.choices(self.buffer[k], k=num))
        return sample

    def sample_dictionary_of_datasets(self, sampling_size):
        dataset_dict = self.sample(sampling_size)
        for k in dataset_dict.keys():
            dataset_dict[k] = tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(dataset_dict[k], dtype=tf.float64)
            )
        return dataset_dict

    def sample_dataset(self, sampling_size):
        data_dict = self.sample(sampling_size)
        datasets = []
        for k in data_dict.keys():
            datasets.append(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(data_dict[k]), dtyp=tf.float64
                )
            )
        dataset = tf.data.Dataset.zip(tuple(datasets))

        return dataset, data_dict.keys()

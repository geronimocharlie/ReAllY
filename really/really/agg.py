import numpy as np
import matplotlib.pyplot as plt
import os, logging

# only print error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Smoothing_aggregator:
    def __init__(self, path, saving_after=100, aggregator_keys=["loss"], max_size=5, init_epoch=0):
        self.aggregator_size = 0
        self.aggregator_max_size = max_size
        self.aggregator = {}
        self.aggregator_vals = {}

        for k in aggregator_keys:
            self.aggregator[k] = []
            self.aggregator_vals[k] = []
        self.keys = aggregator_keys
        self.path = path
        self.saving_after = saving_after
        self.epoch = init_epoch
        self.reached_size = False

    def update(self, **kwargs):
        self.epoch += 1
        increased = False
        saved = False
        for k in kwargs.keys():
            if k in self.aggregator.keys():
                self.aggregator[k].append(kwargs[k])
                #print('agg shape')
                #print(np.asarray(self.aggregator[k]).shape)
                increased = True
            else:
                print(
                    f"unsupported aggregator key: {k}, aggregator was only initialized with the keys {self.aggregator.keys()}"
                )
                raise KeyError

        if increased:
            self.aggregator_size += 1

        if self.aggregator_size >= self.aggregator_max_size:
            for k in kwargs.keys():
                self.aggregator_vals[k].append(np.mean( [np.mean(i) for i in self.aggregator[k]]))
                self.aggregator[k] = []
            self.aggregator_size = 0
            if not (self.reached_size):
                self.reached_size = True

        if (self.epoch % self.saving_after == 0) and self.reached_size:
            if len(self.aggregator_vals[self.keys[0]]) > 1:
                self.save_graphic()

    def save_graphic(self):

        keys = list(self.aggregator_vals.keys())
        number_of_subplots = len(keys)
        plt.clf()
        plt.suptitle(f"training process after {self.epoch} training epochs")
        for i, v in enumerate(range(number_of_subplots)):
            v += 1
            ax1 = plt.subplot(number_of_subplots, 1, v, label=keys[i])
            ax1.plot(self.aggregator_vals[keys[i]])
            ax1.set_ylabel(keys[i])

        plt.xlabel(
            f"1 step aggregated over {self.aggregator_max_size} * aggregated data per epoch"
        )
        plt.savefig(f"{self.path}/progress_{self.epoch}.png")

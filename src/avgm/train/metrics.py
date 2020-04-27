import torch
from collections import defaultdict
import numpy as np


class MultiClassMetrics(object):
    def __init__(self, size):
        self.size = size
        self.current_epoch = None
        self.current_mode = None
        self.data = defaultdict(  # Epoch dictionary
            lambda: defaultdict(  # Mode dictionary
                lambda: np.zeros((self.size, self.size))
            )
        )
        self.name = "acc"

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_mode(self, mode):
        self.current_mode = mode

    def update(self, y_hat, y):
        y_class = y_hat.argmax(axis=1)
        for pred, actual in zip(y_class, y):
            pred, actual = pred.item(), actual.item()
            self.data[self.current_epoch][self.current_mode][pred, actual] += 1

    def calculate(self, epoch=None, mode=None):
        epoch = epoch if epoch is not None else self.current_epoch
        mode = mode if mode is not None else self.current_mode

        cm = self.data[epoch][mode]

        true_pos = np.diag(cm)
        false_pos = cm.sum(axis=0) - true_pos
        false_neg = cm.sum(axis=1) - true_pos
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            accuracy = true_pos.sum() / cm.sum()

        return accuracy, precision, recall

    def __repr__(self):
        acc, _, _ = self.calculate()
        return "{:0.2f}".format(acc)
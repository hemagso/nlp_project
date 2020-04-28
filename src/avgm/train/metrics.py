from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod
import csv


class Metric(ABC):
    def __init__(self):
        self.current_epoch = None
        self.current_mode = None

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_mode(self, mode):
        self.current_mode = mode

    @abstractmethod
    def update(self, y_hat, y):
        pass


class MultiClassMetrics(Metric):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.data = defaultdict(  # Epoch dictionary
            lambda: defaultdict(  # Mode dictionary
                lambda: np.zeros((self.size, self.size))
            )
        )
        self.name = "acc"

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

    def to_csv(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
            writer.writerow(("epoch", "mode", "pred", "actual", "count"))
            for epoch in self.data.keys():
                for mode in self.data[epoch].keys():
                    for i in range(self.size):
                        for j in range(self.size):
                            writer.writerow((epoch, mode, i, j, self.data[epoch][mode][i, j]))

    def __repr__(self):
        acc, _, _ = self.calculate()
        return "{:0.2f}".format(acc)
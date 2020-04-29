from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod
import csv
from typing import Optional, Tuple
import torch


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
    """ This class computes accuracy, precision, recall and confusion matrices for classification.
    It does that by keeping track of the confusion matrix, and only calculating the actual metrics when
    necessary.
    """
    def __init__(self, size: int, input_type: Optional[str] = "scores"):
        """ Creates a new Multi Class Classification  metrics

        :param size: Number of classes
        :param input_type: Whether inputs are Scores (Either raw or probabilities) or the actual classes
        """
        assert input_type in ("scores", "classes")
        super().__init__()
        self.input_type = input_type
        self.size = size
        self.data = defaultdict(  # Epoch dictionary
            lambda: defaultdict(  # Mode dictionary
                lambda: np.zeros((self.size, self.size))
            )
        )
        self.name = "acc"

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        """ Update the metrics with a new batch of observations

        :param y_hat: Predicted values for the class. If they are scores they will be transformed into classes
            by using argmax.
        :param y: Ground truth classes
        """
        if self.input_type == "scores":
            y_hat = y_hat.argmax(axis=1)
        for pred, actual in zip(y_hat, y):
            pred, actual = pred.item(), actual.item()
            self.data[self.current_epoch][self.current_mode][pred, actual] += 1

    def calculate(self, epoch: Optional[int] = None, mode: Optional[str] = None) -> Tuple[float, float, float]:
        """ Calculate the additional metrics.

        :param epoch: Epoch that we want the metrics calculated. If not provided the current one will
            be used (Optional).
        :param mode: Mode that we want the metrics calculated. If not provided the current one will be used (Optional)
        :return: accuracy, precision, recall float tuple.
        """
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

    def to_csv(self, filepath: str) -> None:
        """ Save metrics to a CSV file. All epochs and modes will be saved. Only the confusion matrix data will be
        saved, since other metrics can be calculated from it.

        :param filepath: Path to the file.
        """
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
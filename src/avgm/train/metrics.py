import torch


class MultiClassAccuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.name = "acc"

    def update(self, y_hat, y):
        y_class = y_hat.argmax(axis=1)
        self.correct += (y_class == y).sum().item()
        self.total += y.shape[0]

    def __call__(self):
        return self.correct / self.total


class MAE(object):
    def __init__(self):
        self.sum_error = 0
        self.total = 0
        self.name = "MAE"

    def update(self, y_hat, y):
        y_class = y_hat.argmax(axis=1)
        self.sum_error += torch.abs(y_class - y).sum().item()
        self.total += y.shape[0]

    def __call__(self):
        return self.sum_error / self.total

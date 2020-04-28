import torch.nn as nn
from abc import ABC, abstractmethod


class Callback(ABC):
    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_param_update(self):
        pass


class ClippingCallback(Callback):
    def __init__(self, module: nn.Module, margin: float = 0.0, min_val: float = -1E6) -> None:
        super().__init__()
        self.margin = margin
        self.min_val = min_val
        self.module = module

    def on_param_update(self):
        cutpoints = self.module.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(self.min_val, cutpoints[i+1] - self.margin)

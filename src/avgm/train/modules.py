from typing import Optional
import torch
import torch.nn as nn


class CumulativeLogisticLink(nn.Module):
    def __init__(self, num_classes: int, init_cutpoints: str = 'ordered'):
        assert num_classes > 2, ('Only use this model if you have 3 or more classes')
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        self._init_cutpoints()

    def _init_cutpoints(self):
        num_cutpoints = self.num_classes - 1
        if self.init_cutpoints == 'ordered':
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
        elif self.init_cutpoints == 'random':
            cutpoints = torch.rand(num_cutpoints).sort()[0]
        else:
            raise ValueError('Invalid cutpoint type')
        self.cutpoints = nn.Parameter(cutpoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoids = torch.sigmoid(self.cutpoints - x)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
            sigmoids[:, [0]],
            link_mat,
            (1 - sigmoids[:, [-1]])
        ), dim=1)
        return link_mat


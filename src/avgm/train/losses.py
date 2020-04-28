import torch.nn as nn
from typing import Optional
import torch


class CumulativeLinkLoss(nn.Module):
    def __init__(self, reduction: str = 'elementwise_mean', weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def _reduction(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('Invalid reduction')

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = 1E-15
        y = y.unsqueeze(1)
        likelihoods = torch.clamp(torch.gather(y_hat, 1, y), eps, 1 - eps)
        neg_log_likelihood = -torch.log(likelihoods)

        if self.weight is not None:
            weight = torch.as_tensor(
                self.weight,
                dtype=neg_log_likelihood.dtype,
                device=neg_log_likelihood.device
            )
            neg_log_likelihood *= weight[y]

        return self._reduction(neg_log_likelihood)
import tqdm
import torch

import torch.nn as nn
import torch.optim as optim

from typing import Optional, Iterable, Mapping
from torch.utils import data

from .metrics import Metric
from .callbacks import Callback


class Trainer(object):
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device: str = "cuda",
                 optimizer_kwargs: Optional[Mapping] = None):
        """ Creates an object that handles training multiple epochs of a model

        :param model: Torch module containing the model to be trained
        :param criterion: Torch loss function to be optimize
        :param optimizer: Torch optimizer
        :param device: Device in which we will train the model
        :param optimizer_kwargs: Dict containing optional keyword arguments for the optimizer
        """
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs if optimizer_kwargs is not None else {})
        self.criterion = criterion.to(self.device)

    @staticmethod
    def tqdm(*args, **kwargs) -> tqdm.tqdm:
        """ Utility function to pick the correct version (Notebook or no notebook) of tqdm

        :param args: Positional arguments that will be forwarded to tqdm
        :param kwargs: Keyword arguments that will be forwarded to tqdm
        :return: The proper tqdm object.
        """
        try:
            return tqdm.notebook.tqdm(*args, **kwargs)
        except AttributeError:
            return tqdm.tqdm(*args, **kwargs)

    @staticmethod
    def _update_metrics(metrics: Iterable[Metric], y_hat: torch.Tensor, y: torch.Tensor) -> None:
        """ Update all metrics passed to the run method

        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param y_hat: Tensor containing model predictions
        :param y: Tensor containing actual target values
        """
        for metric in metrics:
            metric.update(y_hat, y)

    @staticmethod
    def _set_metrics_mode(metrics: Iterable[Metric], mode: str) -> None:
        """ Switch metric objects mode. Used to keep metrics from train and evaluation separated from each other.

        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param mode: String containing the mode for the metric. Can be anything, but we use the values 'train' and
            'eval' in this class.
        """
        for metric in metrics:
            metric.set_mode(mode)

    @staticmethod
    def _set_metrics_epoch(metrics: Iterable[Metric], epoch: int) -> None:
        """ Switch epoch for all metric objects. Used to keep metrics from different epochs separated from each other.

        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param epoch: Integer containing the number of the epoch.
        """
        for metric in metrics:
            metric.set_epoch(epoch)

    @staticmethod
    def _fire_callbacks(callbacks: Iterable[Callback], event: str) -> None:
        """ Call a specific method for all callbacks registered.

        :param callbacks: Iterable containing the callbacks to be fired (See callbacks module)
        :param event: String containing the name of the method to be fired.
        """
        for callback in callbacks:
            method = getattr(callback, event)
            method()

    def _run_epoch(self, ds_train: data.DataLoader, ds_valid: Optional[data.DataLoader] = None,
                   pbar_label: str = "", metrics: Iterable[Metric] = (), callbacks: Iterable[Callback] = ()) -> None:
        """ Run a full epoch for the model, both training and, if available, evaluation.

        :param ds_train: Torch Dataloader with the training data
        :param ds_valid: Torch Dataloader with the evaluation data (Optional)
        :param pbar_label: String the the label that will be used on tqdm (Optional)
        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param callbacks: Iterable containing the callbacks to be used during training (See callbacks module)
        """
        label = pbar_label + " (Train)"
        self._run_epoch_train(ds_train, pbar_label=label, metrics=metrics, callbacks=callbacks)
        if ds_valid is not None:
            label = pbar_label + " (Eval)"
            self._run_epoch_eval(ds_valid, pbar_label=label, metrics=metrics, callbacks=callbacks)

    def _run_epoch_train(self, ds_train: data.DataLoader, pbar_label: Optional[str] ="",
                         metrics: Optional[Iterable[Metric]] = (),
                         callbacks: Optional[Iterable[Callback]] = ()) -> None:
        """ Run a full epoch for the training data.

        :param ds_train: Torch Dataloader with the training data
        :param pbar_label: String the the label that will be used on tqdm (Optional)
        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param callbacks: Iterable containing the callbacks to be used during training (See callbacks module)
        """
        self.model.train()
        Trainer._set_metrics_mode(metrics, "train")
        with Trainer.tqdm(total=len(ds_train)) as pbar:
            pbar.set_description(pbar_label)
            epoch_loss = 0
            for idx, (x, y) in enumerate(ds_train):
                x = self._send_to_device(*x)
                y = self._send_to_device(y)

                self.optimizer.zero_grad()
                y_hat = self.model(*x)
                loss = self.criterion(y_hat, y)
                Trainer._update_metrics(metrics, y_hat, y)

                loss.backward()
                self.optimizer.step()
                Trainer._fire_callbacks(callbacks, "on_param_update")

                epoch_loss += loss.item()
                if idx % 25 == 0:  # TODO: Figure out a better printing rule
                    pbar.update(25)
                    pbar.set_postfix(
                        loss=epoch_loss / (idx + 1),
                        refresh=False,
                        **{metric.name: metric for metric in metrics}
                    )

    def _run_epoch_eval(self, ds_valid: data.DataLoader, pbar_label: Optional[str] = "",
                        metrics: Optional[Iterable[Metric]] = (), callbacks: Optional[Iterable[Callback]] = ()) -> None:
        """ Run a full epoch for the evaluation data.

        :param ds_valid: Torch Dataloader with the validation data
        :param pbar_label: String the the label that will be used on tqdm (Optional)
        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param callbacks: Iterable containing the callbacks to be used during training (See callbacks module)
        """
        self.model.eval()
        Trainer._set_metrics_mode(metrics, "eval")
        with Trainer.tqdm(total=len(ds_valid)) as pbar:
            with torch.no_grad():
                pbar.set_description(pbar_label)
                epoch_loss = 0
                for idx, (x, y) in enumerate(ds_valid):
                    x = self._send_to_device(*x)
                    y = self._send_to_device(y)

                    y_hat = self.model(*x)
                    loss = self.criterion(y_hat, y)
                    Trainer._update_metrics(metrics, y_hat, y)

                    epoch_loss += loss.item()
                    if idx % 25 == 0:
                        pbar.update(25)
                        pbar.set_postfix(
                            loss=epoch_loss / (idx + 1),
                            **{metric.name: metric for metric in metrics},
                            refresh=False)

    def run(self, n_epochs: int, ds_train: data.DataLoader, ds_valid: Optional[data.DataLoader] = None,
            metrics: Optional[Iterable[Metric]] = (), callbacks: Optional[Iterable[Callback]] = ()) -> Iterable[Metric]:
        """ Train the model for a specific number of epochs

        :param n_epochs: Number of epochs to be run.
        :param ds_train: Torch Dataloader with the training data
        :param ds_valid: Torch Dataloader with the evaluation data (Optional)
        :param metrics: Iterable containing the metrics to be monitored during training (See metrics module)
        :param callbacks: Iterable containing the callbacks to be used during training (See callbacks module)
        :return: Iterable containing the metrics monitored during training (See metrics module)
        """
        for epoch in range(1, n_epochs + 1):
            Trainer._set_metrics_epoch(metrics, epoch)
            self._run_epoch(
                ds_train,
                ds_valid,
                metrics=metrics,
                callbacks=callbacks,
                pbar_label="Epoch {n}/{total}".format(n=epoch, total=n_epochs)
            )
            Trainer._fire_callbacks(callbacks, "on_epoch_end")
        return metrics

    def _send_to_device(self, *args):
        """ Helper function for sending tensors into a specific device

        :param args: Positional arguments containing all objects that will be sent into a device
        :return: Copy of the tensor on the device
        """
        if len(args) == 1:
            return args[0].to(self.device)
        else:
            return tuple(arg.to(self.device) for arg in args)

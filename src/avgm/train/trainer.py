import tqdm
import torch


class Trainer(object):
    def __init__(self, model, criterion, optimizer, device="cuda", optimizer_kwargs={}):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion.to(self.device)

    @staticmethod
    def tqdm(*args, **kwargs):
        try:
            return tqdm.notebook.tqdm(*args, **kwargs)
        except AttributeError:
            return tqdm.tqdm(*args, **kwargs)

    def _run_epoch(self, ds_train, ds_valid=None, pbar_label="", metrics=(), callbacks=()):
        label = pbar_label + " (Train)"
        self._run_epoch_train(ds_train, pbar_label=label, metrics=metrics, callbacks=callbacks)
        if ds_valid is not None:
            label = pbar_label + " (Eval)"
            self._run_epoch_eval(ds_valid, pbar_label=label, metrics=metrics, callbacks=callbacks)

    @staticmethod
    def _update_metrics(metrics, y_hat, y):
        for metric in metrics:
            metric.update(y_hat, y)

    @staticmethod
    def _set_metrics_mode(metrics, mode):
        for metric in metrics:
            metric.set_mode(mode)

    @staticmethod
    def _set_metrics_epoch(metrics, epoch):
        for metric in metrics:
            metric.set_epoch(epoch)

    @staticmethod
    def _fire_callbacks(callbacks, event):
        for callback in callbacks:
            method = getattr(callback, event)
            method()

    def _run_epoch_train(self, ds_train, pbar_label="", metrics=(), callbacks=()):
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
                if idx % 25 == 0:
                    pbar.update(25)
                    pbar.set_postfix(
                        loss=epoch_loss / (idx + 1),
                        refresh=False,
                        **{metric.name: metric for metric in metrics}
                    )

    def _run_epoch_eval(self, ds_valid, pbar_label="", metrics=(), callbacks=()):
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

    def run(self, n_epochs, ds_train, ds_valid=None, metrics=(), callbacks=()):
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
        if len(args) == 1:
            return args[0].to(self.device)
        else:
            return tuple(arg.to(self.device) for arg in args)

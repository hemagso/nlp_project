import tqdm


class Trainer(object):
    def __init__(self, model, ds_valid=None, device="cuda"):
        self.device = device
        self.optimizer = None
        self.model = None
        self.criterion = None

    def _run_epoch(self, ds_train, ds_valid):
        self.model.train()
        with tqdm.tqdm(total=len(ds_train)) as pbar:
            epoch_loss = 0
            for idx, (x, y) in enumerate(self.ds_train):
                x = self._send_to_device(*x)
                y = self._send_to_device(y)

                self.optimizer.zero_grad()
                y_hat = self.model(*x)
                loss = self.criterion(y_hat, y)

                loss.backward()
                self.optimizer.step()
        if ds_valid is not None:
            pass

    def _send_to_device(self, *args):
        return tuple(arg.to(self.device) for arg in args)

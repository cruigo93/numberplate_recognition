import pytorch_lightning as pl
import datasets
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint


class Learner(pl.LightningModule):
    def __init__(self,
                 dataloaders,
                 model,
                 optimizer,
                 scheduler,
                 config, 
                 encoder) -> None:
        super(Learner, self).__init__()
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        self.encoder = encoder

    def forward(self, x):
        return self.model(x)

    def loss_fn(self, pr, gt):
        bs = gt.size(0)
        # print(gt.shape, pr.shape)
        log_softmax_input = F.log_softmax(pr, 2)
        input_lengths = torch.full(
            size=(bs, ), fill_value=log_softmax_input.size(0), dtype=torch.int32
        )
        target_lengths = torch.full(
            size=(bs, ), fill_value=gt.size(1), dtype=torch.int32
        )
        # print(input_lengths.shape, target_lengths.shape)
        loss = nn.CTCLoss(blank=0)(log_softmax_input, gt,
                                   input_lengths, target_lengths)
        return loss

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def training_step(self, batch, batch_idx):
        x, y, label = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'loss': avg_loss, "step": self.current_epoch}
        # print(avg_loss)
        return {'avg_train_loss': avg_loss,
                "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, label= batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        # print(label)

        return {"val_eval_list": [pred.cpu(), y.cpu(), label], "val_loss": loss}

    def decode(self, preds):
        decoded = []
        preds = preds.permute(1, 0, 2)
        preds = torch.softmax(preds, 2)
        preds = torch.argmax(preds, 2)
        preds = preds.detach().cpu().numpy()
        for i in range(preds.shape[0]):
            temp = []
            for k in preds[i, :]:
                k = k - 1
                if k == -1:
                    temp.append("+")
                else:
                    temp.append(self.encoder.inverse_transform([k])[0])
            decoded.append("".join(temp))
        return decoded
    def print_decoded(self, val_eval_list):
        val_text = []
        labels = []
        for preds, gt, label in val_eval_list:
            curr_text = self.decode(preds)
            val_text.extend(curr_text)
            labels.extend(label)
        pprint(list(zip(val_text, labels))[:5])

    def validation_epoch_end(self, outputs):
        val_eval_list = [x["val_eval_list"] for x in outputs]
        self.print_decoded(val_eval_list)
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_val_loss,
                            "step": self.current_epoch}
        return {"avg_val_loss": avg_val_loss,
                "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, label = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        return {"test_eval_list": [pred.cpu(), y.cpu()], "test_loss": loss}

    def test_epoch_end(self, outputs):
        test_eval_list = [x["test_eval_list"] for x in outputs]
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_test_loss, "step": 1}
        return {"avg_test_loss": avg_test_loss,
                "log": tensorboard_logs}

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

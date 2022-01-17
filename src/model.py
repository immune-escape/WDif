import esm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.metrics import precision_score, recall_score, roc_auc_score



class EscapeMutationModel(pl.LightningModule):
    def __init__(
        self,
        Optimizer=torch.optim.Adam,
        optimizer_kwargs={},
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, batch):
        raise NotImplementedError

    def loss_function(self, outputs, y):
        y = y.long().flatten()
        return nn.CrossEntropyLoss()(outputs, y)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss_function(outputs, batch["label"])

        preds = torch.softmax(outputs, 1)[:, 1].unsqueeze(-1)
        probas = preds.flatten().detach().cpu().numpy()
        target = batch["label"].detach().cpu().numpy().astype(int).flatten()

        try:
            rocauc = roc_auc_score(target, probas)
            precision = precision_score(target, probas > 0.5, zero_division=0)
            recall = recall_score(target, probas > 0.5, zero_division=0)
        except:
            rocauc = np.nan
            precision = np.nan
            recall = np.nan

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/roc_auc", rocauc, on_epoch=True)
        self.log("train/precision", precision, on_epoch=True)
        self.log("train/recall", recall, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.loss_function(outputs, batch["label"])

        preds = torch.softmax(outputs, 1)[:, 1].unsqueeze(-1)
        probas = preds.flatten().detach().cpu().numpy()
        target = batch["label"].detach().cpu().numpy().astype(int).flatten()

        try:
            try:
                rocauc = roc_auc_score(target, probas)
            except ValueError:
                rocauc = 0.5

            precision = precision_score(target, probas > 0.5, zero_division=0)
            recall = recall_score(target, probas > 0.5, zero_division=0)
        except Exception as e:
            print(str(e))
            rocauc = np.nan
            precision = np.nan
            recall = np.nan

        self.log("valid/loss", loss, on_epoch=True)
        self.log("valid/roc_auc", rocauc, on_epoch=True)
        self.log("valid/precision", precision, on_epoch=True)
        self.log("valid/recall", recall, on_epoch=True)
        self.log("hp_metric", rocauc, on_epoch=True)
        return {
            "loss": loss,
            "roc_auc": rocauc,
            "precision": precision,
            "recall": recall,
        }

    def configure_optimizers(self):
        optimizer = self.hparams.Optimizer(
            [params for name, params in self.named_parameters() if "esm" not in name],
            **self.hparams.optimizer_kwargs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[12, 24],
                    gamma=0.1,
                ),
                "monitor": "epoch",
            },
        }

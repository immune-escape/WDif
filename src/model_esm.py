import esm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from model import EscapeMutationModel


class EscapeMutationModelESM(EscapeMutationModel):
    def __init__(
        self,
        reference_protein_seq: str,
        embedding_model=esm.pretrained.esm1_t6_43M_UR50S,
        max_length=1024,
        pre_classifier_num_hidden=128,
        classifier_linear_hiddens=tuple([]),
        ClassifierActivation=nn.ReLU(),
        dropout_rate=0.5,
        use_bn=False,
        Optimizer=torch.optim.Adam,
        optimizer_kwargs={},
        conv_dif=False,
        straight_forward=False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.conv_dif = conv_dif
        self.straight_forward = straight_forward

        embedding_model = self.hparams.embedding_model

        self.esm, self.alphabet = embedding_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        if (
            "msa" in embedding_model.__name__
            or "esm1v" in embedding_model.__name__
            or "esm1b" in embedding_model.__name__
        ):
            self.embedding_size = self.esm.embed_positions.embedding_dim
        else:
            self.embedding_size = self.esm.embed_out.size(1)

        _, _, self.ref_tokens = self.batch_converter(
            [("SARS-CoV-2-WT", self.hparams.reference_protein_seq)]
        )
        self.ref_tokens = self.ref_tokens[:, : self.hparams.max_length]

        if self.conv_dif:
            self.c1 = nn.Conv1d(self.embedding_size, 1, 7, padding=3)
        else:
            # self.linear_0 = nn.Linear(self.hparams.max_length, 1)
            self.theta = nn.Parameter(torch.rand(self.hparams.max_length))

        self.relu = nn.LeakyReLU()

        if self.conv_dif:
            self.pre_classifier = nn.Linear(
                self.hparams.max_length, self.hparams.pre_classifier_num_hidden
            )
        else:
            self.pre_classifier = nn.Linear(
                self.embedding_size, self.hparams.pre_classifier_num_hidden
            )
        self.dropout = nn.Dropout(p=self.hparams.dropout_rate)

        if self.hparams.use_bn:
            self.pre_classifier_bn = nn.BatchNorm1d(
                self.hparams.pre_classifier_num_hidden
            )

        self.final_classifier = nn.Linear(self.hparams.pre_classifier_num_hidden, 2)

        self.ref_embedding = self.embedding(self.ref_tokens.to(self.device)).cpu()

    def embedding(self, x):
        with torch.no_grad():
            outputs = self.esm(x, repr_layers=[self.esm.num_layers])
            return outputs["representations"][self.esm.num_layers]

    def forward(self, batch):
        esm_embedding = self.embedding(batch["inp"])

        ref_embedding = self.ref_embedding.clone().to(self.device)
        if self.conv_dif:
            if self.straight_forward is False:
                ref = self.relu(self.c1(torch.transpose(ref_embedding, 1, 2))).view(
                    -1, self.hparams.max_length
                )
                ref = ref.repeat(batch["inp"].size(0), 1)

            emb = self.relu(self.c1(torch.transpose(esm_embedding, 1, 2))).view(
                -1, self.hparams.max_length
            )
        else:
            if self.straight_forward is False:
                ref = ref_embedding.repeat(batch["inp"].size(0), 1, 1)
            emb = esm_embedding

        if self.straight_forward is False:
            output = ref - emb
        else:
            output = emb

        if self.conv_dif is False:
            output = torch.matmul(torch.transpose(output, 1, 2), self.theta)

        pre_class_logits = self.pre_classifier(output)
        pre_class_logits = self.dropout(pre_class_logits)
        if self.hparams.use_bn:
            pre_class_logits = self.pre_classifier_bn(pre_class_logits)

        logits = self.final_classifier(self.relu(pre_class_logits))

        return logits

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

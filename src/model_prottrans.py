import torch
import torch.nn as nn
import pytorch_lightning as pl
from Bio import SeqIO

from sklearn.metrics import roc_auc_score, precision_score, recall_score
import esm
import numpy as np

import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc

from model import EscapeMutationModel


class EscapeMutationModelProtTrans(EscapeMutationModel):
    def __init__(
        self,
        reference_protein_seq: str,
        embedding_model='Rostlab/prot_t5_xl_uniref50',
        pre_classifier_num_hidden=128,
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

        self.batch_converter = T5Tokenizer.from_pretrained(self.hparams.embedding_model, do_lower_case=False)
        self.embedding_model = T5EncoderModel.from_pretrained(self.hparams.embedding_model)

        ids = self.batch_converter.batch_encode_plus([" ".join(reference_protein_seq)], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])
        with torch.no_grad():
            self.ref_embedding = self.embedding_model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            ).last_hidden_state.cpu()
        self.embedding_size = self.ref_embedding.shape[-1]
        self.max_length = self.ref_embedding.shape[-2]


        if self.conv_dif:
            self.c1 = nn.Conv1d(self.embedding_size, 1, 7, padding=3)
        else:
            self.theta = nn.Parameter(torch.rand(self.max_length))

        self.relu = nn.LeakyReLU()

        if self.conv_dif:
            self.pre_classifier = nn.Linear(
                self.max_length, self.hparams.pre_classifier_num_hidden
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

    def forward(self, batch):
        for k, v in batch.items():
            print(k, v.shape)

        with torch.no_grad():
            inp_embedding = self.embedding_model(
                input_ids=batch['inp'],
                attention_mask=batch['attention_mask'],
            ).last_hidden_state[:, :self.max_length, :]
        

        ref_embedding = self.ref_embedding.clone().to(self.device)

        if self.conv_dif:
            if self.straight_forward is False:
                ref = self.relu(self.c1(torch.transpose(ref_embedding, 1, 2))).view(
                    -1, self.hparams.max_length
                )
                ref = ref.repeat(batch["inp"].size(0), 1)

            emb = self.relu(self.c1(torch.transpose(inp_embedding, 1, 2))).view(
                -1, self.hparams.max_length
            )
        else:
            if self.straight_forward is False:
                ref = ref_embedding.repeat(batch["inp"].size(0), 1, 1)
            emb = inp_embedding

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

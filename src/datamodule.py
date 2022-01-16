import os
import re
from typing import Union

import esm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from Bio import SeqIO
from numpy.lib.arraysetops import isin
from numpy.testing._private.utils import raises
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from transformers.models.t5.tokenization_t5 import T5Tokenizer


def apply_mutations(mutations, seq):
    seq = list(seq)
    if mutations is None:
        return seq

    for idx, mutation in enumerate(mutations):
        pos, t = mutation
        if t.lower() == "del":
            seq[pos] = "!"
        elif t.lower() == "stop":
            seq = seq[:pos]
        elif len(t) > 1:
            pass
        else:
            seq[pos] = t
    for idx, mutation in enumerate(mutations):
        pos, t = mutation
        if len(t) > 1 and t != "del":
            seq = seq[:pos] + list(t) + seq[pos:]

    seq = list(re.sub("!", "", "".join(seq)))
    if len(seq) < 1276:
        seq += ["-"] * (1276 - len(seq))

    return "".join(seq[:1276])


def parse_aaSubstitutions(mutations):
    mutations = mutations.split(",")
    mutations_nf = []
    for mutation in mutations:
        f, t = re.split(r"\d+", mutation)
        pos = int(re.findall(r"\d+", mutation)[0])
        if len(t) == 1 or t == "del":
            mutations_nf.append((pos, t))
        elif "ins" in f:
            mutations_nf.append((pos, t))
        else:
            return None
    return mutations_nf


class ProteinDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return self.df.index.size

    def __getitem__(self, idx):
        inp = self.df["input"].values[idx]
        label = self.df["label"].values[idx]

        return {
            "inp": inp,
            "label": torch.Tensor([label]),
        }


class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_converter: Union[esm.data.BatchConverter, T5Tokenizer],
        reference_protein_seq: str,
        train_data_path="../data/gisaid_ds_v1.1.csv",
        valid_data_path="../data/gisaid_val_ds_v1.1.csv",
        test_data_path="../data/gisaid_test_ds_v1.1.csv",
        batch_size=32,
        num_workers=16,
    ) -> None:
        super().__init__()

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_converter = batch_converter

        self.reference_protein_seq = reference_protein_seq

    def prepare_data(self):
        self.train_df = pd.read_csv(self.train_data_path, parse_dates=True)
        self.train_df["mutations"] = self.train_df["S_aaSubstitutions"].apply(
            parse_aaSubstitutions
        )
        self.train_df = self.train_df.dropna(subset=["S_aaSubstitutions"])

        self.val_df = pd.read_csv(self.valid_data_path, parse_dates=True)
        self.val_df["mutations"] = self.val_df["S_aaSubstitutions"].apply(
            parse_aaSubstitutions
        )
        self.val_df = self.val_df.dropna(subset=["S_aaSubstitutions"])

        self.test_df = pd.read_csv(self.test_data_path, parse_dates=True)
        self.test_df["mutations"] = self.test_df["S_aaSubstitutions"].apply(
            parse_aaSubstitutions
        )
        self.test_df = self.test_df.dropna(subset=["S_aaSubstitutions"])

    def setup(self, stage=None):
        if stage == "test":
            self.test_df = self._preprocess_df(self.test_df)
        else:
            self.train_df = self._preprocess_df(self.train_df)
            self.val_df = self._preprocess_df(self.val_df)

    def train_dataloader(self):
        return DataLoader(
            ProteinDataset(self.train_df),
            batch_size=self.batch_size,
            # drop_last=True,
            num_workers=self.num_workers,
            sampler=torch.utils.data.WeightedRandomSampler(
                self.train_df["weights"],
                self.train_df.index.size,
                replacement=True,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            ProteinDataset(self.val_df),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            ProteinDataset(self.test_df),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _preprocess_df(self, df):
        df = df.reset_index()
        df["label"] = df["label"].apply(int)
        df["name"] = df["S_aaSubstitutions"].apply(lambda x: "id_" + str(x))
        df["seq"] = df["mutations"].apply(
            lambda muts: apply_mutations(muts, self.reference_protein_seq)
        )
        df["input"] = df.apply(lambda x: (x["name"], str(x["seq"])), axis=1)

        train_targets = torch.tensor(df["label"].values.astype(int))
        _, counts = np.unique(train_targets, return_counts=True)
        class_weights = np.array([sum(counts) / c for c in counts])

        target_list = torch.tensor(df["label"].values.astype(int))

        weights = class_weights[target_list]

        df["weights"] = weights
        return df

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x = batch["inp"]
        if isinstance(self.batch_converter, esm.data.BatchConverter):
            _, _, batch_tokens = self.batch_converter(list(zip(x[0], x[1])))
            batch["inp"] = batch_tokens[:, :1024]
        elif isinstance(self.batch_converter, T5Tokenizer):
            ids = self.batch_converter.batch_encode_plus(
                [" ".join(seq) for seq in batch["inp"][1]],
                add_special_tokens=True,
                padding=True,
            )

            input_ids = torch.tensor(ids["input_ids"])[:, :1275]
            attention_mask = torch.tensor(ids["attention_mask"])[:1275]

            batch["inp"] = input_ids
            batch["attention_mask"] = attention_mask
        else:
            raise TypeError(
                f"batch_converter must be T5Tokenizer or BatchConverter type, but "
                f"{type(self.batch_converter)} provided"
            )

        if len(batch) > 1:
            batch["label"] = batch["label"].float()
        return batch

import itertools
from ctypes import Union
from pathlib import Path

import esm
import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from Bio import SeqIO
from torch.optim import optimizer

from datamodule import ProteinDataModule
from model_esm import EscapeMutationModelESM
from model_prottrans import EscapeMutationModelProtTrans


OPTIMIZERS = {
    "RMSprop": torch.optim.RMSprop,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

EMBEDDING_MODELS = {
    "esm1_t6_43M_UR50S": (EscapeMutationModelESM, esm.pretrained.esm1_t6_43M_UR50S),
    "esm1b_t33_650M_UR50S": (EscapeMutationModelESM, esm.pretrained.esm1b_t33_650M_UR50S),
    "esm1_t12_85M_UR50S": (EscapeMutationModelESM, esm.pretrained.esm1_t12_85M_UR50S),
    "esm1_t34_670M_UR50S": (EscapeMutationModelESM, esm.pretrained.esm1_t34_670M_UR50S),
    "Rostlab/prot_t5_xl_uniref50": (EscapeMutationModelProtTrans, "Rostlab/prot_t5_xl_uniref50"),
}


def fit(
    conv_dif=False,
    straight_forward=False,
    batch_size=16,
    run_path="./runs/",
    optimizer="RMSprop",
    lr=1e-3,
    weight_decay=0.10,
    embedding_model="Rostlab/prot_t5_xl_uniref50",
    gpus=1,
    max_epochs=100,
    reference_protein_seq_fasta_path="./data/cov2_spike_wt.fasta",
    train_data_path="./data/gisaid_ds_v1.1.csv",
    valid_data_path="./data/gisaid_val_ds_v1.1.csv",
    test_data_path="./data/gisaid_test_ds_v1.1.csv",
):
    run_optimizer = OPTIMIZERS.get(optimizer)
    if run_optimizer is None:
        raise ValueError(
            f"optimizer must be on of {OPTIMIZERS.keys()}, but {optimizer} passed"
        )

    RunModel, run_embedding = EMBEDDING_MODELS.get(embedding_model, (None, None))
    if RunModel is None or run_embedding is None:
        raise ValueError(
            f"optimizer must be on of {EMBEDDING_MODELS.keys()}, but {embedding_model} passed"
        )

    with open(reference_protein_seq_fasta_path, "r") as f:
        reference_protein_record = [r for r in SeqIO.parse(f, "fasta")][0]

    run_path = (
        Path(run_path)
        / f"SF_{straight_forward}"
        / f"conv_dif_{conv_dif}"
        / RunModel.__name__
        / str(run_embedding)
        / run_optimizer.__name__
    )
    run_path = run_path / f"wd_{weight_decay}__lr_{lr}__bs_{batch_size}"

    model = RunModel(
        reference_protein_seq=reference_protein_record.seq,
        embedding_model=run_embedding,
        Optimizer=run_optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        conv_dif=conv_dif,
        straight_forward=straight_forward,
    )

    dm = ProteinDataModule(
        reference_protein_seq=reference_protein_record.seq,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        batch_converter=model.batch_converter,
        batch_size=batch_size,
    )

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        default_root_dir=str(run_path),
        deterministic=True,
        log_every_n_steps=20,
    )

    trainer.fit(model, datamodule=dm)

    preds = trainer.predict(model, dm.val_dataloader())
    vdf = dm.val_df
    vdf["preds"] = torch.softmax(torch.cat(preds), 1)[:, 1].cpu().detach().numpy()
    vdf.to_csv(run_path / "val_preds.csv", index=False)

    dm.setup("test")
    preds = trainer.predict(model, dm.test_dataloader())
    vdf = dm.test_df
    vdf["preds"] = torch.softmax(torch.cat(preds), 1)[:, 1].cpu().detach().numpy()
    vdf.to_csv(run_path / "test_preds.csv", index=False)


if __name__ == "__main__":
    fire.Fire(fit)

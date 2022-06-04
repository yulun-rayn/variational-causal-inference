# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch

from evaluate.evaluate import evaluate

from dpi.model import DPI

from dataset.dataset import load_dataset_splits, data_collate

from utils.general_utils import pjson


def prepare_dpi(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_dataset_splits(
        args["data"],
        args["perturbation_key"],
        args["control_key"],
        args["dose_key"],
        args["covariate_keys"],
        args["split_key"],
        args["dist_mode"],
    )

    autoencoder = DPI(
        datasets["training"].num_genes,
        datasets["training"].num_perturbations,
        datasets["training"].num_covariates,
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        patience=args["patience"],
        dist_mode=args["dist_mode"],
        hparams=args["hparams"]
    )
    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def train_dpi(args, return_model=False):
    """
    Trains a DPI autoencoder
    """

    autoencoder, datasets = prepare_dpi(args)

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=autoencoder.hparams["batch_size"],
                shuffle=True,
                collate_fn=data_collate
            )
        }
    )

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})
    args["hparams"] = autoencoder.hparams

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_dir = os.path.join(args["artifact_path"], 'saves/' + args["name"] + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            (genes, perts, cf_genes, cf_perts, covariates) = (
            data[0], data[1], data[2], data[3], data[4:])

            minibatch_training_stats = autoencoder.update(
                genes, perts, cf_genes, cf_perts, covariates
            )

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(epoch_training_stats[key])
        autoencoder.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or (
            epoch == args["max_epochs"] - 1
        )

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate(autoencoder, datasets)
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history["stats_epoch"].append(epoch)

            pjson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            pjson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            stop = stop or autoencoder.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return autoencoder, datasets

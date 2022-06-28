import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch

from evaluate.evaluate import evaluate

from model.model import PotentialOutcomeVI

from dataset.dataset import load_dataset_splits

from utils.general_utils import pjson
from utils.data_utils import data_collate

def prepare(args, state_dict=None):
    """
    Instantiates model and dataset to run an experiment.
    """

    device = torch.device("cpu") if args["cpu"] else torch.device(
        'cuda:' + str(args["gpu"]) if torch.cuda.is_available() else "cpu"
    )

    datasets = load_dataset_splits(
        args["data"],
        args["perturbation_key"],
        args["control_key"],
        args["dose_key"],
        args["covariate_keys"],
        args["split_key"],
        True if args["dist_mode"] == 'match' else False,
    )

    model = PotentialOutcomeVI(
        datasets["training"].num_genes,
        datasets["training"].num_perturbations,
        datasets["training"].num_covariates,
        outcome_dist=args["outcome_dist"],
        dist_mode=args["dist_mode"],
        patience=args["patience"],
        seed=args["seed"],
        device=device,
        hparams=args["hparams"]
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, datasets


def train(args, return_model=False):
    """
    Trains a VCI model
    """

    model, datasets = prepare(args)

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=model.hparams["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    pjson({"training_args": args})
    pjson({"model_params": model.hparams})
    args["hparams"] = model.hparams

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_dir = os.path.join(args["artifact_path"], 'saves/' + args["name"] + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            (genes, perts, cf_genes, cf_perts, covariates) = (
            data[0], data[1], data[2], data[3], data[4:])

            minibatch_training_stats = model.update(
                genes, perts, cf_genes, cf_perts, covariates
            )

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_training_stats[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = (
            (epoch == args["max_epochs"] - 1) 
            #or (ellapsed_minutes > args["max_minutes"])
        )

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate(model, datasets)
            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            pjson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (model.state_dict(), args, model.history),
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
            stop = stop or model.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return model, datasets

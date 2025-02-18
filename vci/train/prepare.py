import torch

from ..model.model import load_VCI
from ..model.classifier import load_classifier
from ..dataset.dataset import load_dataset_splits
from ..utils.data_utils import data_collate

def prepare(args, state_dict=None, device="cuda"):
    """
    Instantiates model and dataset to run an experiment.
    """

    # dataset
    datasets = load_dataset_splits(
        args["data_name"], args["data_path"],
        label_names=(args["label_names"].split(",") if args["label_names"] is not None else None),
        sample_cf=(True if args["dist_mode"] == "match" else False),
    )

    datasets.update(
        {
            "train_loader": torch.utils.data.DataLoader(
                datasets["train"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=datasets["train"].nb_dims))
            )
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    args["num_covariates"] = datasets["train"].num_covariates

    # model
    model = load_VCI(args, state_dict, device)

    args["hparams"] = model.hparams

    return model, datasets

def prepare_classifier(args, state_dict=None, device="cuda"):
    """
    Instantiates model and dataset to run an experiment.
    """

    # dataset
    datasets = load_dataset_splits(
        args["data_name"], args["data_path"],
        label_names=(args["label_names"].split(",") if args["label_names"] is not None else None),
        sample_cf=False,
    )

    datasets.update(
        {
            "train_loader": torch.utils.data.DataLoader(
                datasets["train"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=datasets["train"].nb_dims))
            ),
            "test_loader": torch.utils.data.DataLoader(
                datasets["test"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=datasets["test"].nb_dims))
            ),
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    args["num_covariates"] = datasets["train"].num_covariates

    # model
    model = load_classifier(args, state_dict, device)

    args["hparams"] = model.hparams

    return model, datasets

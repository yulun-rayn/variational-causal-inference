import os
import time
import logging
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from .prepare import prepare, prepare_classifier

from ..evaluate.evaluate import evaluate, evaluate_loss
from ..utils.general_utils import initialize_logger, ljson
from ..utils.data_utils import move_tensors

def train(args, prepare=prepare, evaluate=evaluate):
    """
    Trains a VCI model
    """
    state_dict = None
    if args["checkpoint"] is not None:
        state_dict, args = torch.load(args["checkpoint"], map_location="cpu")

    device = args["device"] if torch.cuda.is_available() else "cpu"

    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    model, datasets = prepare(args, state_dict=state_dict, device=device)

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

    log_dir = os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt)
    writer = SummaryWriter(log_dir=log_dir)
    initialize_logger(log_dir)

    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    os.makedirs(save_dir, exist_ok=True)

    ljson({"training_args": args})
    logging.info("")
    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for batch_idx, batch in enumerate(datasets["train_loader"]):
            minibatch_training_stats = model.update(
                move_tensors(*batch, device=device), batch_idx, writer
            )

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val
        model.step()

        ellapsed_minutes = (time.time() - start_time) / 60

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["train_loader"])

        # decay learning rate if necessary
        # also check stopping condition: 
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats, early_stop = evaluate(model, datasets,
                epoch=epoch, save_dir=save_dir, **args
            )

            ljson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )
            for key, val in epoch_training_stats.items():
                writer.add_scalar(key, val, epoch)

            torch.save(
                (model.state_dict(), args),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )
            ljson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )

            if stop:
                ljson({"stop": epoch})
                break
            if early_stop:
                ljson({"early_stop": epoch})
                break

    writer.close()
    return model

def train_classifier(args, prepare=prepare_classifier, evaluate=evaluate_loss):
    return train(args, prepare=prepare, evaluate=evaluate)

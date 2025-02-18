import copy
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import MLP

from ..utils.model_utils import total_grad_norm_

#####################################################
#                     LOAD MODEL                    #
#####################################################

def load_classifier(args, state_dict=None, device="cuda"):
    if args["data_name"] == "gene":
        model = Classifier(
            args["num_outcomes"],
            args["num_treatments"],
            args["num_covariates"],
            device=device,
            hparams=args["hparams"]
        )
    elif args["data_name"] in ("celebA", "morphoMNIST"):
        model = ClassifierConv(
            args["num_outcomes"],
            args["num_treatments"],
            args["num_covariates"],
            device=device,
            hparams=args["hparams"]
        )
    else:
        raise ValueError("data_name not recognized")

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

#####################################################
#                     MAIN MODEL                    #
#####################################################

class Classifier(nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        device="cuda",
        hparams=None
    ):
        super().__init__()
        # generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        # early-stopping
        self.best_score = -np.inf
        self.patience_trials = 0

        # set hyperparameters
        self._set_hparams(hparams)

        self._init_model()

        self.iteration = 0

        self.to_device(device)

    def _set_hparams(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "classifier_resolution": "64,32,16,8,4,1",
            "classifier_width": "32,64,128,256,512,1024",
            "classifier_depth": "2,3,3,2,1,2",
            "classifier_fc_width": 1024,
            "classifier_fc_depth": 2,
            "classifier_lr": 3e-4,
            "classifier_wd": 4e-7,
            "classifier_ss": 500,
            "max_grad_norm": -1,
            "grad_skip_threshold": -1,
            "patience": 20,
        }

        # the user may fix some hparams
        if hparams is not None:
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        return self.hparams

    def _init_model(self):
        self.model = self.init_model()

        # optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["classifier_lr"],
            weight_decay=self.hparams["classifier_wd"],
        )
        self.sch = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=self.hparams["classifier_ss"]
        )

    def forward(self, outcomes, covariates):
        return self.model(outcomes, covariates).squeeze()

    def loss(self, outcomes, treatments, covariates):
        predicts = self.forward(outcomes, covariates)

        loss = F.mse_loss(predicts, treatments)
        return loss, {"MSE": loss.item()}

    def update(self, batch, batch_idx=-1, writer=None):
        outcomes, treatments, covariates, _, _ = batch

        loss, loss_log = self.loss(outcomes, treatments, covariates)

        self.opt.zero_grad()
        loss.backward()
        if self.hparams["max_grad_norm"] > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams["max_grad_norm"])
        else:
            grad_norm = total_grad_norm_(self.model.parameters())
        if self.hparams["grad_skip_threshold"] < 0 or grad_norm < self.hparams["grad_skip_threshold"]:
            self.opt.step()

        if writer is not None:
            writer.add_scalar("Grad Norm", grad_norm.item(), self.iteration)

        self.iteration += 1

        return loss_log

    def evaluate(self, batch, batch_idx=-1):
        outcomes, treatments, covariates, _, _ = batch

        with torch.autograd.no_grad():
            _, loss_log = self.loss(outcomes, treatments, covariates)

        return loss_log

    def step(self):
        self.sch.step()

    def early_stopping(self, score=None):
        if score is None:
            return None

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.hparams["patience"]

    def init_model(self):
        return MLP([self.num_outcomes]
            + [self.hparams["classifier_width"]] * (self.hparams["classifier_depth"] - 1)
            + [self.num_treatments+self.num_covariates]
        )

    def to_device(self, device):
        self.device = device
        self.to(self.device)

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for VCI
        """

        return self._set_hparams(self, "")

#####################################################
#                     EXTENSIONS                    #
#####################################################

from .convolution import ConvModel

from ..utils.model_utils import parse_block_string


class ClassifierConv(Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_hparams(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "classifier_resolution": "64,32,16,8,4,1",
            "classifier_width": "32,64,128,256,512,1024",
            "classifier_depth": "2,3,3,2,1,2",
            "classifier_fc_width": 1024,
            "classifier_fc_depth": 2,
            "classifier_lr": 3e-4,
            "classifier_wd": 4e-7,
            "classifier_ss": 500,
            "max_grad_norm": -1,
            "grad_skip_threshold": -1,
            "patience": 20,
        }

        # the user may fix some hparams
        if hparams is not None:
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        return self.hparams

    def init_model(self):
        return ConvModel(
            *parse_block_string(
                self.hparams["classifier_resolution"],
                self.hparams["classifier_width"],
                self.hparams["classifier_depth"],
                in_size=self.num_outcomes,
                out_size=(self.num_treatments, 1, 1)
            ),
            num_features=sum(self.num_covariates),
            lite_blocks=True, lite_layers=True
        )

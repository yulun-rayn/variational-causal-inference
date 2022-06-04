import json

import torch
import torch.nn.functional as F

from dpi.module import MLP

from utils.math_utils import (
    logprob_normal,
    logprob_nb_positive,
    logprob_zinb_positive
)

#####################################################
#                    MODEL SAVING                   #
#####################################################

def init_DPI(state):
    net = DPI(state['input_dim'],
                state['nb_hidden'],
                state['nb_layers'],
                state['output_dim'])
    net.load_state_dict(state['state_dict'])
    return net

def load_DPI(state_path):
    state = torch.load(state_path)
    return init_DPI(state)

def save_DPI(net, state_path=None):
    torch.save(net.get_dict(), state_path)

#####################################################
#                     MAIN MODEL                    #
#####################################################

class DPI(torch.nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=False,
        embed_covariates=True,
        device="cuda",
        seed=0,
        patience=5,
        loss_ae="normal",
        dist_mode='match',
        aggr='concat',
        hparams=""
    ):
        super(DPI, self).__init__()
        # set generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        self.dist_mode=dist_mode
        self.aggr=aggr
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hparams)

        if self.loss_ae == 'nb':
            self.num_dist_params = 2
        elif self.loss_ae == 'zinb':
            self.num_dist_params = 3
        elif self.loss_ae == 'normal':
            self.num_dist_params = 2
        else:
            raise ValueError("loss_ae not recognized")

        params = []

        if self.embed_outcomes:
            self.outcomes_embeddings = MLP(
                [num_outcomes, self.hparams["encoder_width"]]
            )
            outcome_dim = self.hparams["encoder_width"]
            params.extend(list(self.outcomes_embeddings.parameters()))
        else:
            outcome_dim = num_outcomes

        if self.embed_treatments:
            self.treatments_embeddings = torch.nn.Embedding(
                self.num_treatments, self.hparams["encoder_width"]
            )
            treatment_dim = self.hparams["encoder_width"]
            params.extend(list(self.treatments_embeddings.parameters()))
        else:
            treatment_dim = num_treatments

        if self.embed_covariates:
            covariates_emb_dim = self.hparams["encoder_width"]//len(self.num_covariates)
            self.covariates_embeddings = []
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, 
                        covariates_emb_dim
                    )
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )
            covariate_dim = covariates_emb_dim*len(self.num_covariates)
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))
        else:
            covariate_dim = sum(num_covariates)

        # set models
        self.encoder = MLP(
            [outcome_dim+2*treatment_dim+covariate_dim]
            + [self.hparams["encoder_width"]] * self.hparams["encoder_depth"]
            + [self.hparams["dim"]]
        )
        params.extend(list(self.encoder.parameters()))

        self.decoder = MLP(
            [self.hparams["dim"]]
            + [self.hparams["decoder_width"]] * self.hparams["decoder_depth"]
            + [num_outcomes * self.num_dist_params]
        )
        params.extend(list(self.decoder.parameters()))

        # optimizer
        self.optimizer_autoencoder = torch.optim.Adam(
            params,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        if self.dist_mode == 'classify':
            self.treatment_classifier = MLP(
                [self.hparams["dim"]]
                + [self.hparams["classifier_width"]] * self.hparams["classifier_depth"]
                + [num_treatments]
            )
            self.loss_treatment_classifier = torch.nn.CrossEntropyLoss()
            params = list(self.treatment_classifier.parameters())

            self.covariate_classifier = []
            self.loss_covariate_classifier = []
            for num_covariate in self.num_covariates:
                adv = MLP(
                    [num_outcomes]
                    + [self.hparams["classifier_width"]]
                        * self.hparams["classifier_depth"]
                    + [num_covariate]
                )
                self.covariate_classifier.append(adv)
                self.loss_covariate_classifier.append(torch.nn.CrossEntropyLoss())
                params.extend(list(adv.parameters()))

            self.optimizer_classifier = torch.optim.Adam(
                params,
                lr=self.hparams["classifier_lr"],
                weight_decay=self.hparams["classifier_wd"],
            )
            self.scheduler_classifier = torch.optim.lr_scheduler.StepLR(
                self.optimizer_classifier, step_size=self.hparams["step_size_lr"]
            )
        elif self.dist_mode == 'estimate':
            self.outcome_estimator = MLP(
                [treatment_dim+covariate_dim]
                + [self.hparams["estimator_width"]] * self.hparams["estimator_depth"]
                + [num_outcomes * self.num_dist_params]
            )
            self.loss_outcome_estimator = torch.nn.CrossEntropyLoss()
            params = list(self.outcome_estimator.parameters())

            self.optimizer_estimator = torch.optim.Adam(
                params,
                lr=self.hparams["estimator_lr"],
                weight_decay=self.hparams["estimator_wd"],
            )
            self.scheduler_estimator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_adversaries, step_size=self.hparams["step_size_lr"]
            )
        elif self.dist_mode == 'match':
            pass
        else:
            raise ValueError("dist_mode not recognized")

        self.iteration = 0

        self.to(self.device)

        self.history = {"epoch": [], "stats_epoch": []}

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "dim": 128,
            "dosers_width": 128,
            "dosers_depth": 2,
            "encoder_width": 128,
            "encoder_depth": 2,
            "decoder_width": 128,
            "decoder_depth": 2,
            "classifier_width": 64,
            "classifier_depth": 2,
            "estimator_width": 64,
            "estimator_depth": 2,
            "reg_dist": 1.5,
            "autoencoder_lr": 3e-4,
            "classifier_lr": 3e-4,
            "estimator_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "classifier_wd": 4e-7,
            "estimator_wd": 4e-7,
            "batch_size": 64,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs(self, outcomes, treatments, cf_treatments, covariates):
        """
        Move minibatch tensors to CPU/GPU.
        """
        outcomes = outcomes.to(self.device)
        treatments = treatments.to(self.device)
        cf_treatments = cf_treatments.to(self.device)
        if covariates is not None:
            covariates = [cov.to(self.device) for cov in covariates]
        return (outcomes, treatments, cf_treatments, covariates)

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def predict(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        eps=1e-3
    ):
        """
        Predict "what would have the gene expression `outcomes` been, had the
        cells in `outcomes` with cell types `cell_types` been treated with
        combination of treatments `treatments`.
        """ 

        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )

        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))
            cf_treatments = self.treatments_embeddings(cf_treatments.argmax(1))
        if self.embed_covariates:
            for i, emb in enumerate(self.covariates_embeddings):
                covariates[i] = emb(covariates[i].argmax(1))

        latent = self.encoder(torch.cat([outcomes, treatments, cf_treatments] + covariates, 1))
        outcomes_re = self.decoder(latent)

        if self.loss_ae == 'nb':
            nb_mus = F.softplus(outcomes_re[:, :self.num_outcomes]).add(eps)
            nb_thetas = F.softplus(outcomes_re[:, self.num_outcomes:]).add(eps)
            outcomes_re = (nb_mus, nb_thetas)
        elif self.loss_ae == 'zinb':
            nb_mus = F.softplus(outcomes_re[:, :self.num_outcomes]).add(eps)
            nb_thetas = F.softplus(outcomes_re[:, self.num_outcomes:(2*self.num_outcomes)]).add(eps)
            zi_logits = outcomes_re[:, (2*self.num_outcomes):].add(eps)
            outcomes_re = (nb_mus, nb_thetas, zi_logits)
        elif self.loss_ae == 'normal':
            normal_locs = outcomes_re[:, :self.num_outcomes].add(eps)
            normal_scales = F.softplus(outcomes_re[:, self.num_outcomes:]).add(eps)
            outcomes_re = (normal_locs, normal_scales)

        return outcomes_re

    def loss(self, outcomes, outcomes_re):
        """
        Compute loss.
        """
        num = len(outcomes)
        weights = None
        if isinstance(outcomes, list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_re = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_re
            ]
            outcomes = torch.cat(outcomes, 0)

        if self.loss_ae == 'nb':
            loss = -logprob_nb_positive(outcomes,
                mu=outcomes_re[0],
                theta=outcomes_re[1],
                weight=weights
            )
        elif self.loss_ae == 'zinb':
            loss = -logprob_zinb_positive(outcomes,
                mu=outcomes_re[0],
                theta=outcomes_re[1],
                zi_logits=outcomes_re[2],
                weight=weights
            )
        elif self.loss_ae == 'normal':
            loss = -logprob_normal(outcomes,
                loc=outcomes_re[0],
                scale=outcomes_re[1],
                weight=weights
            )

        return (loss.sum(0)/num).mean()

    def update(self, outcomes, treatments, cf_outcomes, cf_treatments, covariates):
        """
        Update DPI's parameters given a minibatch of outcomes, treatments, and
        cell types.
        """
        if cf_outcomes is None:
            assert self.dist_mode != 'match'
        else:
            cf_outcomes = [out.to(self.device) for out in cf_outcomes]
        outcomes = outcomes.to(self.device)
        treatments = treatments.to(self.device)
        cf_treatments = cf_treatments.to(self.device)
        covariates = [cov.to(self.device) for cov in covariates]

        outcomes_re = self.predict(
            outcomes,
            treatments,
            treatments,
            covariates
        )
        outcomes_cf = self.predict(
            outcomes,
            treatments,
            cf_treatments,
            covariates
        )

        reconstruction_loss = self.loss(outcomes, outcomes_re)
        if self.dist_mode == 'classify':
            raise NotImplementedError(
                "TODO: implement dist_mode 'classify' for distribution loss")
        elif self.dist_mode == 'estimate':
            raise NotImplementedError(
                "TODO: implement dist_mode 'estimate' for distribution loss")
        elif self.dist_mode == 'match':
            distribution_loss = self.loss(cf_outcomes, outcomes_cf)

        loss = reconstruction_loss + self.hparams["reg_dist"]*distribution_loss
        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_distribution": distribution_loss.item()
        }

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for DPI
        """

        return self.set_hparams_(self, "")

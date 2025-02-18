import copy
import json
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .distribution import Bernoulli, NegativeBinomial, ZeroInflatedNegativeBinomial
from .module import MLP
from .classifier import load_classifier

from ..utils.math_utils import (
    logprob_normal, kldiv_normal,
    logprob_bernoulli_logits,
    logprob_nb_positive,
    logprob_zinb_positive
)
from ..utils.model_utils import lr_lambda_exp, lr_lambda_lin, total_grad_norm_

#####################################################
#                     LOAD MODEL                    #
#####################################################

def load_VCI(args, state_dict=None, device="cuda"):
    if args["dist_mode"] == "classify":
        assert args["checkpoint_classifier"] is not None
        state_dict_classifier, args_classifier = torch.load(
            args["checkpoint_classifier"], map_location="cpu")

        classifier = load_classifier(args_classifier, state_dict_classifier)
        classifier.eval()
    else:
        classifier = None

    if args["data_name"] == "gene":
        model = VCI(
            args["num_outcomes"],
            args["num_treatments"],
            args["num_covariates"],
            embed_outcomes=True,
            embed_treatments=False,
            embed_covariates=True,
            omega0=args["omega0"],
            omega1=args["omega1"],
            omega2=args["omega2"],
            dist_outcomes=args["dist_outcomes"],
            dist_mode=args["dist_mode"],
            classifier=classifier,
            mc_sample_size=30,
            lr_lambda=lr_lambda_exp,
            device=device,
            hparams=args["hparams"]
        )
    elif args["data_name"] == "celebA":
        model = HVCIConv(
            args["num_outcomes"],
            args["num_treatments"],
            args["num_covariates"],
            embed_outcomes=True,
            embed_treatments=True,
            embed_covariates=False,
            omega0=args["omega0"],
            omega1=args["omega1"],
            omega2=args["omega2"],
            dist_outcomes=args["dist_outcomes"],
            dist_mode=args["dist_mode"],
            classifier=classifier,
            mc_sample_size=3,
            lr_lambda=lr_lambda_exp,
            device=device,
            hparams=args["hparams"]
        )
    elif args["data_name"] == "morphoMNIST":
        model = HVCIConv(
            args["num_outcomes"],
            args["num_treatments"],
            args["num_covariates"],
            embed_outcomes=True,
            embed_treatments=True,
            embed_covariates=False,
            omega0=args["omega0"],
            omega1=args["omega1"],
            omega2=args["omega2"],
            dist_outcomes=args["dist_outcomes"],
            dist_mode=args["dist_mode"],
            classifier=classifier,
            mc_sample_size=3,
            lr_lambda=(lambda e: lr_lambda_lin(args["max_epochs"], fixed_epochs=e)),
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

class VCI(nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=True,
        embed_covariates=True,
        omega0=1.0,
        omega1=2.0,
        omega2=0.1,
        dist_outcomes="normal",
        dist_mode="match",
        classifier=None,
        mc_sample_size=30,
        lr_lambda=lr_lambda_exp,
        device="cuda",
        hparams=None
    ):
        super().__init__()
        # generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.dist_outcomes = dist_outcomes
        self.mc_sample_size = mc_sample_size
        self.lr_lambda = lr_lambda
        # vci parameters
        self.omega0 = omega0
        self.omega1 = omega1
        self.omega2 = omega2
        self.dist_mode = dist_mode
        self.classifier = classifier
        # early-stopping
        self.best_score = -np.inf
        self.patience_trials = 0

        # set hyperparameters
        self._set_hparams(hparams)

        # individual-specific model
        self._init_indiv_model()

        # covariate-specific model
        self._init_covar_model()

        self.iteration = 0

        self.to_device(device)

    def _set_hparams(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "outcome_emb_dim": 256,
            "treatment_emb_dim": 64,
            "covariate_emb_dim": 16,
            "latent_dim": 128,
            "encoder_width": 128,
            "encoder_depth": 3,
            "decoder_width": 128,
            "decoder_depth": 3,
            "discriminator_width": 64,
            "discriminator_depth": 3,
            "generator_lr": 3e-4,
            "generator_wd": 4e-7,
            "discriminator_lr": 3e-4,
            "discriminator_wd": 4e-7,
            "discriminator_freq": 2,
            "opt_step_size": 400,
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

        self.outcome_dim = (
            self.hparams["outcome_emb_dim"] if self.embed_outcomes else self.num_outcomes)
        self.treatment_dim = (
            self.hparams["treatment_emb_dim"] if self.embed_treatments else self.num_treatments)
        self.covariate_dim = (
            self.hparams["covariate_emb_dim"] * len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )

        return self.hparams

    def _init_indiv_model(self):
        # embeddings
        outcomes_embeddings = self.init_outcome_emb()
        treatments_embeddings = self.init_treatment_emb()
        covariates_embeddings = self.init_covariates_emb()

        # models
        encoder = self.init_encoder()
        decoder = self.init_decoder()

        self.generator = nn.ModuleDict({
            "outcomes_embeddings": outcomes_embeddings,
            "treatments_embeddings": treatments_embeddings,
            "covariates_embeddings": covariates_embeddings,
            "encoder": encoder,
            "decoder": decoder
        })
        self.encoder_eval = copy.deepcopy(encoder)

        # optimizer
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams["generator_lr"],
            weight_decay=self.hparams["generator_wd"],
        )
        self.g_sch = torch.optim.lr_scheduler.LambdaLR(
            self.g_opt, lr_lambda=self.lr_lambda(self.hparams["opt_step_size"])
        )

    def _init_covar_model(self):
        if self.dist_mode == "classify":
            assert self.classifier is not None
        elif self.dist_mode == "discriminate":
            # embeddings
            outcomes_embeddings = self.init_outcome_emb()
            treatments_embeddings = self.init_treatment_emb()
            covariates_embeddings = self.init_covariates_emb()

            # model
            discriminator = self.init_discriminator()

            self.discriminator = nn.ModuleDict({
                "outcomes_embeddings": outcomes_embeddings,
                "treatments_embeddings": treatments_embeddings,
                "covariates_embeddings": covariates_embeddings,
                "discriminator": discriminator
            })

            # optimizer
            self.d_opt = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.hparams["discriminator_lr"],
                weight_decay=self.hparams["discriminator_wd"],
            )
            self.d_sch = torch.optim.lr_scheduler.LambdaLR(
                self.d_opt, lr_lambda=self.lr_lambda(self.hparams["opt_step_size"])
            )
        elif self.dist_mode == "match":
            pass
        else:
            raise ValueError("dist_mode not recognized")

    def encode(self, outcomes, treatments, covariates,
               distributionize=True, dist="normal", evaluate=False):
        outcomes = self.generator["outcomes_embeddings"](outcomes)
        treatments = self.generator["treatments_embeddings"](treatments)
        covariates = [emb(covars) for covars, emb in 
            zip(covariates, self.generator["covariates_embeddings"])
        ]

        if evaluate:
            out = self.encoder_eval(outcomes, treatments, covariates)
        else:
            out = self.generator["encoder"](outcomes, treatments, covariates)

        if distributionize:
            return self.distributionize(out, dist=dist)
        return out

    def decode(self, latents, treatments,
               distributionize=True, dist=None):
        treatments = self.generator["treatments_embeddings"](treatments)

        out = self.generator["decoder"](latents, treatments)

        if distributionize:
            return self.distributionize(out, dist=dist)
        return out

    def discriminate(self, outcomes, treatments, covariates):
        outcomes = self.discriminator["outcomes_embeddings"](outcomes)
        treatments = self.discriminator["treatments_embeddings"](treatments)
        covariates = [emb(covars) for covars, emb in 
            zip(covariates, self.discriminator["covariates_embeddings"])
        ]

        return self.discriminator["discriminator"](outcomes, treatments, covariates).squeeze()

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param sigma: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def distributionize(self, constructions, dist=None, eps=1e-3):
        if dist is None:
            dist = self.dist_outcomes

        if dist == "nb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            dist = NegativeBinomial(
                mu=mus, theta=thetas
            )
        elif dist == "zinb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            zi_logits = constructions[..., 2].add(eps)
            dist = ZeroInflatedNegativeBinomial(
                mu=mus, theta=thetas, zi_logits=zi_logits
            )
        elif dist == "normal":
            locs = constructions[..., 0]
            scales = F.softplus(constructions[..., 1]).add(eps)
            dist = Normal(
                loc=locs, scale=scales
            )
        elif dist == "bernoulli":
            logits = constructions[..., 0]
            dist = Bernoulli(
                logits=logits
            )

        return dist

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor, treatments: torch.Tensor, 
            size: int = 1, distributionize: bool = True, dist: str = None) -> torch.Tensor:
        mu = mu.repeat(size, *[1]*(mu.ndim-1))
        sigma = sigma.repeat(size, *[1]*(sigma.ndim-1))
        treatments = treatments.repeat(size, *[1]*(treatments.ndim-1))

        latents = self.reparameterize(mu, sigma)

        return self.decode(latents, treatments, distributionize=distributionize, dist=dist)

    def predict(
        self,
        outcomes,
        treatments,
        covariates,
        cf_treatments,
        return_dist=False
    ):
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_dist = self.encode(outcomes, treatments, covariates)

            outcomes_dist = self.decode(latents_dist.mean, cf_treatments)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean

    def generate(
        self,
        outcomes,
        treatments,
        covariates,
        cf_treatments,
        return_dist=False
    ):
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_dist = self.encode(outcomes, treatments, covariates)

            outcomes_dist_samp = self.decode(latents_dist.sample(), cf_treatments)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def logprob(self, outcomes, outcomes_param, dist=None):
        """
        Compute log likelihood.
        """
        if dist is None:
            dist = self.dist_outcomes

        num = len(outcomes)
        if isinstance(outcomes, list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes],
                device=outcomes[0].device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_param = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_param
            ]
            outcomes = torch.cat(outcomes, 0)
        elif isinstance(outcomes_param[0], list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes_param[0]],
                device=outcomes_param[0][0].device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes = torch.repeat_interleave(outcomes, sizes, dim=0)
            outcomes_param = [
                torch.cat(out, 0)
                for out in outcomes_param
            ]
        else:
            weights = None

        if dist == "nb":
            logprob = logprob_nb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                weight=weights
            )
        elif dist == "zinb":
            logprob = logprob_zinb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                zi_logits=outcomes_param[2],
                weight=weights
            )
        elif dist == "normal":
            logprob = logprob_normal(outcomes,
                loc=outcomes_param[0],
                scale=outcomes_param[1],
                weight=weights
            )
        elif dist == "bernoulli":
            logprob = logprob_bernoulli_logits(outcomes,
                loc=outcomes_param[0],
                weight=weights
            )

        return (logprob.sum(0)/num).mean()

    def forward(self, outcomes, treatments, covariates, cf_treatments,
                sample_latent=True, sample_outcome=False,
                detach_encode=False, detach_eval=False):
        """
        Execute the workflow.
        """

        # q(z | y, x, t)
        latents_dist = self.encode(outcomes, treatments, covariates)
        if detach_eval:
            latents_dist_eval = self.encode(
                outcomes, treatments, covariates, evaluate=True)
        else:
            latents_dist_eval = latents_dist

        # p(y | z, t)
        outcomes_dist_samp = self.sample(
            latents_dist.mean, latents_dist.stddev, treatments, size=self.mc_sample_size)

        # p(y' | z, t')
        if sample_latent:
            cf_outcomes_dist_out = self.decode(latents_dist.rsample(), cf_treatments)
        else:
            cf_outcomes_dist_out = self.decode(latents_dist.mean, cf_treatments)
        if sample_outcome:
            cf_outcomes_out = cf_outcomes_dist_out.rsample()
        else:
            cf_outcomes_out = cf_outcomes_dist_out.mean

        # q(z | y', x, t')
        if detach_encode:
            if sample_latent:
                cf_outcomes_dist_in = self.decode(latents_dist.sample(), cf_treatments)
            else:
                cf_outcomes_dist_in = self.decode(latents_dist.mean.detach(), cf_treatments)
            if sample_outcome:
                cf_outcomes_in = cf_outcomes_dist_in.rsample()
            else:
                cf_outcomes_in = cf_outcomes_dist_in.mean
        else:
            cf_outcomes_in = cf_outcomes_out
        cf_latents_dist_eval = self.encode(
            cf_outcomes_in, cf_treatments, covariates, evaluate=detach_eval)

        return (outcomes_dist_samp, cf_outcomes_out, latents_dist_eval, cf_latents_dist_eval)

    def loss_reconstruction(self, outcomes_dist, outcomes):
        return -outcomes_dist.log_prob(outcomes).mean()

    def loss_causality(self, cf_outcomes_out, cf_treatments, covariates, cf_outcomes=None,
                       hinge_threshold=0.05, kde_kernel_std=1.0):
        if self.dist_mode == "classify":
            classifier_loss, _ = self.classifier.loss(cf_outcomes_out, cf_treatments, covariates)
            if hinge_threshold is not None:
                return F.relu(classifier_loss - hinge_threshold) + hinge_threshold
            return classifier_loss
        elif self.dist_mode == "discriminate":
            return F.softplus(-self.discriminate(cf_outcomes_out, cf_treatments, covariates)).mean()
        elif self.dist_mode == "match":
            notNone = [o != None for o in cf_outcomes]
            cf_outcomes = [o for (o, n) in zip(cf_outcomes, notNone) if n]
            cf_outcomes_out = cf_outcomes_out[notNone]

            kernel_std = [kde_kernel_std * torch.ones_like(o) for o in cf_outcomes]
            return -self.logprob(cf_outcomes_out, (cf_outcomes, kernel_std), dist="normal")

    def loss_disentanglement(self, latents_dist, cf_latents_dist):
        return kldiv_normal(
            latents_dist.mean, latents_dist.stddev,
            cf_latents_dist.mean, cf_latents_dist.stddev
        ).mean()

    def loss(self, outcomes, treatments, covariates, cf_treatments, cf_outcomes=None):
        """
        Compute losses.
        """
        outcomes_dist_samp, cf_outcomes_out, latents_dist, cf_latents_dist = self.forward(
            outcomes, treatments, covariates, cf_treatments
        )

        # (1) individual-specific likelihood
        indiv_spec_nllh = self.loss_reconstruction(
            outcomes_dist_samp,
            outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        )

        # (2) covariate-specific likelihood
        covar_spec_nllh = self.loss_causality(
            cf_outcomes_out, cf_treatments, covariates, cf_outcomes=cf_outcomes
        )

        # (3) kl divergence
        kl_divergence = self.loss_disentanglement(latents_dist, cf_latents_dist)

        return (self.omega0 * indiv_spec_nllh
            + self.omega1 * covar_spec_nllh
            + self.omega2 * kl_divergence
        ), {"Indiv-spec NLLH": indiv_spec_nllh.item(),
            "Covar-spec NLLH": covar_spec_nllh.item(),
            "KL Divergence": kl_divergence.item()
        }

    def loss_discriminator(self, outcomes, treatments, covariates, cf_treatments):
        cf_outcomes = self.generate(outcomes, treatments, covariates, cf_treatments)

        score_real = self.discriminate(outcomes, treatments, covariates)
        score_fake = self.discriminate(cf_outcomes.detach(), cf_treatments, covariates)

        loss_real = F.softplus(-score_real).mean()
        loss_fake = F.softplus(score_fake).mean()

        return (loss_real+loss_fake)/2., {
            "Real Sample Loss": loss_real.item(),
            "Fake Sample Loss": loss_fake.item()
        }

    def update(self, batch, batch_idx=-1, writer=None):
        outcomes, treatments, covariates, cf_treatments, cf_outcomes = batch

        loss_log = {}

        if self.dist_mode == "discriminate":
            if (batch_idx+1) % self.hparams["discriminator_freq"] == 0:
                d_loss, d_log = self.loss_discriminator(outcomes, treatments, covariates, cf_treatments)

                self.d_opt.zero_grad()
                d_loss.backward()
                if self.hparams["max_grad_norm"] > 0:
                    d_grad_norm = nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.hparams["max_grad_norm"])
                else:
                    d_grad_norm = total_grad_norm_(self.discriminator.parameters())
                if self.hparams["grad_skip_threshold"] < 0 or d_grad_norm < self.hparams["grad_skip_threshold"]:
                    self.d_opt.step()

                loss_log.update(d_log)
                if writer is not None:
                    writer.add_scalar("Discriminator Grad Norm", d_grad_norm.item(), self.iteration)

        g_loss, g_log = self.loss(outcomes, treatments, covariates, cf_treatments, cf_outcomes)

        self.g_opt.zero_grad()
        g_loss.backward()
        if self.hparams["max_grad_norm"] > 0:
            g_grad_norm = nn.utils.clip_grad_norm_(self.generator.parameters(), self.hparams["max_grad_norm"])
        else:
            g_grad_norm = total_grad_norm_(self.generator.parameters())
        if self.hparams["grad_skip_threshold"] < 0 or g_grad_norm < self.hparams["grad_skip_threshold"]:
            self.g_opt.step()

        loss_log.update(g_log)
        if writer is not None:
            writer.add_scalar("Generator Grad Norm", g_grad_norm.item(), self.iteration)

        self.iteration += 1

        return loss_log

    def step(self):
        if self.dist_mode == "discriminate":
            self.d_sch.step()
        self.g_sch.step()

        for target_param, param in zip(
            self.encoder_eval.parameters(), self.generator["encoder"].parameters()
        ):
            target_param.data.copy_(param.data)

    def early_stopping(self, score=None):
        if score is None:
            return False

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.hparams["patience"]

    def init_outcome_emb(self):
        if self.embed_outcomes:
            return MLP([self.num_outcomes, self.hparams["outcome_emb_dim"]])
        else:
            return nn.Identity()

    def init_treatment_emb(self):
        if self.embed_treatments:
            return MLP([self.num_treatments, self.hparams["treatment_emb_dim"]])
        else:
            return nn.Identity()

    def init_covariates_emb(self):
        if self.embed_covariates:
            covariates_emb = []
            for num_cov in self.num_covariates:
                covariates_emb.append(
                    MLP([num_cov, self.hparams["covariate_emb_dim"]])
                )
            return nn.ModuleList(covariates_emb)
        else:
            return nn.ModuleList([nn.Identity()]*len(self.num_covariates))

    def init_encoder(self):
        return MLP([self.outcome_dim+self.treatment_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_dim"]],
            heads=2, final_act="relu"
        )

    def init_decoder(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["latent_dim"]+self.treatment_dim]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )

    def init_discriminator(self):
        return MLP([self.outcome_dim+self.treatment_dim+self.covariate_dim]
            + [self.hparams["discriminator_width"]] * (self.hparams["discriminator_depth"] - 1)
            + [1]
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

from ..utils.model_utils import conv_1x1, conv_3x3, parse_block_string


class VCIConv(VCI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_hparams(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "outcome_emb_dim": 32,
            "treatment_emb_dim": 8,
            "covariate_emb_dim": 2,
            "encoder_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
            "encoder_width": "32,64,128,256,512,1024",
            "encoder_depth": "3,12,12,6,3,3",
            "decoder_resolution": "1*1,4*4,8*8,16*16,32*32,64*64",
            "decoder_width": "1024,512,256,128,64,32",
            "decoder_depth": "3,3,6,12,12,3",
            "discriminator_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
            "discriminator_width": "32,64,128,256,512,1024",
            "discriminator_depth": "3,12,12,6,3,9",
            "generator_lr": 0.0003,
            "generator_wd": 4e-07,
            "discriminator_lr": 0.0003,
            "discriminator_wd": 4e-07,
            "discriminator_freq": 2,
            "opt_step_size": 400,
            "max_grad_norm": -1,
            "grad_skip_threshold": -1,
            "patience": 20
        }

        # the user may fix some hparams
        if hparams is not None:
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        self.outcome_dim = (
            (self.hparams["outcome_emb_dim"], *self.num_outcomes[1:]) 
            if self.embed_outcomes else self.num_outcomes)
        self.treatment_dim = (
            self.hparams["treatment_emb_dim"] if self.embed_treatments else self.num_treatments)
        self.covariate_dim = (
            self.hparams["covariate_emb_dim"] * len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )

        return self.hparams

    def init_outcome_emb(self):
        if self.embed_outcomes:
            return conv_1x1(
                self.num_outcomes[0],
                self.hparams["outcome_emb_dim"],
                len(self.num_outcomes)-1
            )
        else:
            return nn.Identity()

    def init_encoder(self):
        return ConvModel(
            *parse_block_string(
                self.hparams["encoder_resolution"],
                self.hparams["encoder_width"],
                self.hparams["encoder_depth"],
                in_size=self.outcome_dim
            ),
            num_features=self.treatment_dim+self.covariate_dim, heads=2,
            lite_blocks=True, lite_layers=True
        )

    def init_decoder(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return ConvModel(
            *parse_block_string(
                self.hparams["decoder_resolution"],
                self.hparams["decoder_width"],
                self.hparams["decoder_depth"],
                out_size=self.num_outcomes
            ),
            num_features=self.treatment_dim, heads=heads,
            lite_blocks=False, lite_layers=True
        )

    def init_discriminator(self):
        return ConvModel(
            *parse_block_string(
                self.hparams["discriminator_resolution"],
                self.hparams["discriminator_width"],
                self.hparams["discriminator_depth"],
                in_size=self.outcome_dim,
                out_size=(1, 1, 1)
            ),
            num_features=self.treatment_dim+self.covariate_dim,
            lite_blocks=True, lite_layers=True, spectral_norm=True
        )


from .hierarchy import HConvEncoder, HConvDecoder


class HVCIConv(VCIConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_hparams(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "outcome_emb_dim": 32,
            "treatment_emb_dim": 8,
            "covariate_emb_dim": 2,
            "defuse_steps": 3,
            "encoder_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
            "encoder_width": "32,64,128,256,512,1024",
            "encoder_depth": "3,12,12,6,3,3",
            "decoder_resolution": "1*1,4*4,8*8,16*16,32*32,64*64",
            "decoder_width": "1024,512,256,128,64,32",
            "decoder_depth": "3,3,6,12,12,3",
            "discriminator_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
            "discriminator_width": "32,64,128,256,512,1024",
            "discriminator_depth": "3,12,12,6,3,9",
            "generator_lr": 0.0003,
            "generator_wd": 4e-07,
            "discriminator_lr": 0.0003,
            "discriminator_wd": 4e-07,
            "discriminator_freq": 2,
            "opt_step_size": 400,
            "max_grad_norm": -1,
            "grad_skip_threshold": -1,
            "patience": 20
        }

        # the user may fix some hparams
        if hparams is not None:
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        self.outcome_dim = (
            (self.hparams["outcome_emb_dim"], *self.num_outcomes[1:]) 
            if self.embed_outcomes else self.num_outcomes)
        self.treatment_dim = (
            self.hparams["treatment_emb_dim"] if self.embed_treatments else self.num_treatments)
        self.covariate_dim = (
            self.hparams["covariate_emb_dim"] * len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )

        return self.hparams

    def encode(self, outcomes, treatments, covariates,
               distributionize=True, dist="normal", evaluate=False):
        outcomes = self.generator["outcomes_embeddings"](outcomes)
        treatments = self.generator["treatments_embeddings"](treatments)
        covariates = [emb(covars) for covars, emb in 
            zip(covariates, self.generator["covariates_embeddings"])
        ]

        if evaluate:
            outs, hiddens = self.encoder_eval(outcomes, treatments, covariates)
        else:
            outs, hiddens = self.generator["encoder"](outcomes, treatments, covariates)

        if distributionize:
            return [self.distributionize(out, dist=dist) for out in outs], hiddens
        return outs, hiddens

    def decode(self, latents, treatments,
               distributionize=True, dist=None):
        treatments = self.generator["treatments_embeddings"](treatments)

        out, hiddens = self.generator["decoder"](latents, treatments)

        if distributionize:
            return self.distributionize(out, dist=dist), hiddens
        return out, hiddens

    def sample(self, mu, sigma, treatments,
               size=1, distributionize=True, dist=None):
        mu = [m.repeat(size, *[1]*(m.ndim-1)) for m in mu]
        sigma = [s.repeat(size, *[1]*(s.ndim-1)) for s in sigma]
        treatments = treatments.repeat(size, *[1]*(treatments.ndim-1))

        latents = [self.reparameterize(m, s) for m, s in zip(mu, sigma)]

        return self.decode(latents, treatments, distributionize=distributionize, dist=dist)

    def predict(
        self,
        outcomes,
        treatments,
        covariates,
        cf_treatments,
        return_dist=False
    ):
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_dist, _ = self.encode(outcomes, treatments, covariates)

            outcomes_dist, _ = self.decode([d.mean for d in latents_dist], cf_treatments)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean

    def generate(
        self,
        outcomes,
        treatments,
        covariates,
        cf_treatments,
        return_dist=False
    ):
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_dist, _ = self.encode(outcomes, treatments, covariates)

            outcomes_dist_samp, _ = self.decode([d.sample() for d in latents_dist], cf_treatments)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def forward(self, outcomes, treatments, covariates, cf_treatments,
                sample_latent=True, sample_outcome=False,
                detach_encode=False, detach_eval=True):
        """
        Execute the workflow.
        """

        # q(z | y, x, t)
        latents_dist, hiddens_in = self.encode(outcomes, treatments, covariates)
        if detach_eval:
            latents_dist_eval, _ = self.encode(
                outcomes, treatments, covariates, evaluate=True)
        else:
            latents_dist_eval = latents_dist

        # p(y | z, t)
        outcomes_dist_samp, hiddens_out_samp = self.sample(
            [d.mean for d in latents_dist], [d.stddev for d in latents_dist],
            treatments, size=self.mc_sample_size)

        # p(y' | z, t')
        if sample_latent:
            cf_outcomes_dist_out, cf_hiddens_out = self.decode(
                [d.rsample() for d in latents_dist], cf_treatments)
        else:
            cf_outcomes_dist_out, cf_hiddens_out = self.decode(
                [d.mean for d in latents_dist], cf_treatments)
        if sample_outcome:
            cf_outcomes_out = cf_outcomes_dist_out.rsample()
        else:
            cf_outcomes_out = cf_outcomes_dist_out.mean

        # q(z | y', x, t')
        if detach_encode:
            if sample_latent:
                cf_outcomes_dist_in, _ = self.decode(
                    [d.sample() for d in latents_dist], cf_treatments)
            else:
                cf_outcomes_dist_in, _ = self.decode(
                    [d.mean.detach() for d in latents_dist], cf_treatments)
            if sample_outcome:
                cf_outcomes_in = cf_outcomes_dist_in.rsample()
            else:
                cf_outcomes_in = cf_outcomes_dist_in.mean
        else:
            cf_outcomes_in = cf_outcomes_out

        cf_latents_dist, cf_hiddens_in = self.encode(cf_outcomes_in, cf_treatments, covariates)
        if detach_eval:
            cf_latents_dist_eval, _ = self.encode(
                cf_outcomes_in, cf_treatments, covariates, evaluate=detach_eval)
        else:
            cf_latents_dist_eval = cf_latents_dist

        return (
            outcomes_dist_samp, cf_outcomes_out, latents_dist_eval, cf_latents_dist_eval,
            hiddens_in, hiddens_out_samp, cf_hiddens_in, cf_hiddens_out
        )

    def loss_reconstruction(self, outcomes_dist_samp, outcomes,
                            hiddens_out_samp, hiddens_in, hiddens_ratio=0.5):
        indiv_spec_nllh = super().loss_reconstruction(outcomes_dist_samp, outcomes)

        for h_out, h_in in zip(hiddens_out_samp, reversed(hiddens_in)):
            indiv_spec_nllh = (indiv_spec_nllh + 
                hiddens_ratio * F.mse_loss(h_out, h_in.detach())
            )
        return indiv_spec_nllh

    def loss_causality(self, cf_outcomes_out, cf_treatments, covariates,
                       cf_hiddens_in, cf_hiddens_out, cf_outcomes=None,
                       hiddens_ratio=0.5, hinge_threshold=0.05, kde_kernel_std=1):
        covar_spec_nllh = super().loss_causality(
            cf_outcomes_out, cf_treatments, covariates, cf_outcomes, hinge_threshold, kde_kernel_std
        )

        for h_in, h_out in zip(cf_hiddens_in, reversed(cf_hiddens_out)):
            covar_spec_nllh = (covar_spec_nllh + 
                hiddens_ratio * F.mse_loss(h_in, h_out.detach())
            )
        return covar_spec_nllh

    def loss_disentanglement(self, latents_dist, cf_latents_dist):
        kl_divs = [
            kldiv_normal(dist.mean, dist.stddev, cf_dist.mean, cf_dist.stddev).mean()
            for dist, cf_dist in zip(latents_dist, cf_latents_dist)
        ]
        return sum(kl_divs)

    def loss(self, outcomes, treatments, covariates, cf_treatments, cf_outcomes=None):
        """
        Compute losses.
        """
        (
            outcomes_dist_samp, cf_outcomes_out, latents_dist, cf_latents_dist,
            hiddens_in, hiddens_out_samp, cf_hiddens_in, cf_hiddens_out
        ) = self.forward(outcomes, treatments, covariates, cf_treatments)

        # (1) individual-specific likelihood
        indiv_spec_nllh = self.loss_reconstruction(
            outcomes_dist_samp,
            outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1)),
            hiddens_out_samp,
            [hidden.repeat(self.mc_sample_size, *[1]*(hidden.dim()-1)) for hidden in hiddens_in]
        )

        # (2) covariate-specific likelihood
        covar_spec_nllh = self.loss_causality(
            cf_outcomes_out, cf_treatments, covariates,
            cf_hiddens_in, cf_hiddens_out,
            cf_outcomes=cf_outcomes
        )

        # (3) kl divergence
        kl_divergence = self.loss_disentanglement(latents_dist, cf_latents_dist)

        return (self.omega0 * indiv_spec_nllh
            + self.omega1 * covar_spec_nllh
            + self.omega2 * kl_divergence
        ), {"Indiv-spec NLLH": indiv_spec_nllh.item(),
            "Covar-spec NLLH": covar_spec_nllh.item(),
            "KL Divergence": kl_divergence.item()
        }

    def init_encoder(self):
        return HConvEncoder(
            *parse_block_string(
                self.hparams["encoder_resolution"],
                self.hparams["encoder_width"],
                self.hparams["encoder_depth"],
                in_size=self.outcome_dim
            ),
            num_features=self.treatment_dim+self.covariate_dim, heads=2,
            defuse_steps=self.hparams["defuse_steps"]
        )

    def init_decoder(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return HConvDecoder(
            *parse_block_string(
                self.hparams["decoder_resolution"],
                self.hparams["decoder_width"],
                self.hparams["decoder_depth"],
                out_size=self.num_outcomes
            ),
            num_features=self.treatment_dim, heads=heads,
            infuse_steps=self.hparams["defuse_steps"]
        )

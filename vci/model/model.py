import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .module import MLP, Bernoulli, NegativeBinomial, ZeroInflatedNegativeBinomial

from ..utils.model_utils import CompoundEmbedding
from ..utils.math_utils import (
    logprob_normal,
    kldiv_normal,
    logprob_bernoulli_logits,
    logprob_nb_positive,
    logprob_zinb_positive
)

#####################################################
#                     LOAD MODEL                    #
#####################################################

def load_VCI(args, state_dict=None):
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    model = VCI(
        args["num_outcomes"],
        args["num_treatments"],
        args["num_covariates"],
        omega0=args["omega0"],
        omega1=args["omega1"],
        omega2=args["omega2"],
        dist_mode=args["dist_mode"],
        dist_outcomes=args["dist_outcomes"],
        patience=args["patience"],
        device=device,
        hparams=args["hparams"]
    )
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
        embed_treatments=False,
        embed_covariates=True,
        omega0=1.0,
        omega1=2.0,
        omega2=0.1,
        dist_mode="match",
        dist_outcomes="normal",
        type_treatments=None,
        type_covariates=None,
        mc_sample_size=30,
        best_score=-1e3,
        patience=5,
        device="cuda",
        hparams=""
    ):
        super(VCI, self).__init__()
        # generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.dist_outcomes = dist_outcomes
        self.type_treatments = type_treatments
        self.type_covariates = type_covariates
        self.mc_sample_size = mc_sample_size
        # vci parameters
        self.omega0 = omega0
        self.omega1 = omega1
        self.omega2 = omega2
        self.dist_mode = dist_mode
        # early-stopping
        self.best_score = best_score
        self.patience = patience
        self.patience_trials = 0

        # set hyperparameters
        self._set_hparams_(hparams)

        # individual-specific model
        self._init_indiv_model()

        # covariate-specific model
        self._init_covar_model()

        self.iteration = 0

        self.history = {"epoch": [], "stats_epoch": []}

        self.to_device(device)

    def _set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "latent_dim": 128,
            "outcome_emb_dim": 256,
            "treatment_emb_dim": 64,
            "covariate_emb_dim": 16,
            "encoder_width": 128,
            "encoder_depth": 3,
            "decoder_width": 128,
            "decoder_depth": 3,
            "discriminator_width": 64,
            "discriminator_depth": 2,
            "autoencoder_lr": 3e-4,
            "discriminator_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "discriminator_wd": 4e-7,
            "discriminator_steps": 3,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
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
            self.hparams["covariate_emb_dim"]*len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )

        return self.hparams

    def _init_indiv_model(self):
        params = []

        # embeddings
        if self.embed_outcomes:
            self.outcomes_embeddings = self.init_outcome_emb()
            params.extend(list(self.outcomes_embeddings.parameters()))

        if self.embed_treatments:
            self.treatments_embeddings = self.init_treatment_emb()
            params.extend(list(self.treatments_embeddings.parameters()))

        if self.embed_covariates:
            self.covariates_embeddings = nn.Sequential(*self.init_covariates_emb())
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))

        # models
        self.encoder = self.init_encoder()
        params.extend(list(self.encoder.parameters()))

        self.encoder_eval = copy.deepcopy(self.encoder)

        self.decoder = self.init_decoder()
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

        return self.encoder, self.decoder

    def _init_covar_model(self):

        if self.dist_mode == "discriminate":
            params = []

            # embeddings
            if self.embed_outcomes:
                self.adv_outcomes_emb = self.init_outcome_emb()
                params.extend(list(self.adv_outcomes_emb.parameters()))

            if self.embed_treatments:
                self.adv_treatments_emb = self.init_treatment_emb()
                params.extend(list(self.adv_treatments_emb.parameters()))

            if self.embed_covariates:
                self.adv_covariates_emb = nn.Sequential(*self.init_covariates_emb())
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))

            # model
            self.discriminator = self.init_discriminator()
            self.loss_discriminator = nn.BCEWithLogitsLoss()
            params.extend(list(self.discriminator.parameters()))

            self.optimizer_discriminator = torch.optim.Adam(
                params,
                lr=self.hparams["discriminator_lr"],
                weight_decay=self.hparams["discriminator_wd"],
            )
            self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=self.hparams["step_size_lr"]
            )

            return self.discriminator

        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')

        elif self.dist_mode == "match":
            return None

        else:
            raise ValueError("dist_mode not recognized")

    def encode(self, outcomes, treatments, covariates, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        if eval:
            return self.encoder_eval(inputs)
        else:
            return self.encoder(inputs)

    def decode(self, latents, treatments):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)

        inputs = torch.cat([latents, treatments], -1)

        return self.decoder(inputs)

    def discriminate(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.adv_outcomes_emb(outcomes)
        if self.embed_treatments:
            treatments = self.adv_treatments_emb(treatments)
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.adv_covariates_emb)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        return self.discriminator(inputs).squeeze()

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

    def distributionize(self, constructions, dim=None, dist=None, eps=1e-3):
        if dim is None:
            dim = self.num_outcomes
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
            size=1) -> torch.Tensor:
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        treatments = treatments.repeat(size, 1)

        latents = self.reparameterize(mu, sigma)

        return self.decode(latents, treatments)

    def predict(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr = self.decode(latents_dist.mean, cf_treatments)
            outcomes_dist = self.distributionize(outcomes_constr)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean

    def generate(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr_samp = self.sample(
                latents_dist.mean, latents_dist.stddev, cf_treatments
            )
            outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

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
                [out.size(0) for out in outcomes], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_param = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_param
            ]
            outcomes = torch.cat(outcomes, 0)
        elif isinstance(outcomes_param[0], list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes_param[0]], device=self.device
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

    def loss(self, outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, cf_treatments,
            covariates, kde_kernel_std=1.0):
        """
        Compute losses.
        """
        # (1) individual-specific likelihood
        indiv_spec_nllh = -outcomes_dist_samp.log_prob(
            outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        ).mean()

        # (2) covariate-specific likelihood
        if self.dist_mode == "discriminate":
            if self.iteration % self.hparams["discriminator_steps"]:
                self.update_discriminator(
                    outcomes, cf_outcomes_out.detach(),
                    treatments, cf_treatments, covariates
                )

            covar_spec_nllh = self.loss_discriminator(
                self.discriminate(cf_outcomes_out, cf_treatments, covariates),
                torch.ones(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
            )
        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')
        elif self.dist_mode == "match":
            notNone = [o != None for o in cf_outcomes]
            cf_outcomes = [o for (o, n) in zip(cf_outcomes, notNone) if n]
            cf_outcomes_out = cf_outcomes_out[notNone]

            kernel_std = [kde_kernel_std * torch.ones_like(o) 
                for o in cf_outcomes]
            covar_spec_nllh = -self.logprob(
                cf_outcomes_out, (cf_outcomes, kernel_std), dist="normal"
            )

        # (3) kl divergence
        kl_divergence = kldiv_normal(
            latents_dist.mean,
            latents_dist.stddev,
            cf_latents_dist.mean,
            cf_latents_dist.stddev
        )

        return (indiv_spec_nllh, covar_spec_nllh, kl_divergence)

    def forward(self, outcomes, treatments, cf_treatments, covariates,
                sample_latent=True, sample_outcome=False, detach_encode=False, detach_eval=True):
        """
        Execute the workflow.
        """

        # q(z | y, x, t)
        latents_constr = self.encode(outcomes, treatments, covariates)
        latents_dist = self.distributionize(
            latents_constr, dim=self.hparams["latent_dim"], dist="normal"
        )

        # p(y | z, t)
        outcomes_constr_samp = self.sample(latents_dist.mean, latents_dist.stddev,
            treatments, size=self.mc_sample_size
        )
        outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        # p(y' | z, t')
        if sample_latent:
            cf_outcomes_constr_out = self.decode(latents_dist.rsample(), cf_treatments)
        else:
            cf_outcomes_constr_out = self.decode(latents_dist.mean, cf_treatments)
        if sample_outcome:
            cf_outcomes_out = self.distributionize(cf_outcomes_constr_out).rsample()
        else:
            cf_outcomes_out = self.distributionize(cf_outcomes_constr_out).mean

        # q(z | y', x, t')
        if detach_encode:
            if sample_latent:
                cf_outcomes_constr_in = self.decode(latents_dist.sample(), cf_treatments)
            else:
                cf_outcomes_constr_in = self.decode(latents_dist.mean.detach(), cf_treatments)
            if sample_outcome:
                cf_outcomes_in = self.distributionize(cf_outcomes_constr_in).rsample()
            else:
                cf_outcomes_in = self.distributionize(cf_outcomes_constr_in).mean
        else:
            cf_outcomes_in = cf_outcomes_out

        cf_latents_constr = self.encode(
            cf_outcomes_in, cf_treatments, covariates, eval=detach_eval
        )
        cf_latents_dist = self.distributionize(
            cf_latents_constr, dim=self.hparams["latent_dim"], dist="normal"
        )

        return (outcomes_dist_samp, cf_outcomes_out,latents_dist, cf_latents_dist)

    def update(self, outcomes, treatments, cf_outcomes, cf_treatments, covariates):
        """
        Update VCI's parameters given a minibatch of outcomes, treatments, and covariates.
        """
        outcomes, treatments, cf_outcomes, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_outcomes, cf_treatments, covariates
        )

        outcomes_dist_samp, cf_outcomes_out, latents_dist, cf_latents_dist = self.forward(
            outcomes, treatments, cf_treatments, covariates
        )

        indiv_spec_nllh, covar_spec_nllh, kl_divergence = self.loss(
            outcomes, outcomes_dist_samp, cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist, treatments, cf_treatments, covariates
        )

        loss = (self.omega0 * indiv_spec_nllh
            + self.omega1 * covar_spec_nllh
            + self.omega2 * kl_divergence
        )

        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1

        return {
            "Indiv-spec NLLH": indiv_spec_nllh.item(),
            "Covar-spec NLLH": covar_spec_nllh.item(),
            "KL Divergence": kl_divergence.item()
        }

    def update_discriminator(self, outcomes, cf_outcomes_out,
                                treatments, cf_treatments, covariates):
        loss_tru = self.loss_discriminator(
            self.discriminate(outcomes, treatments, covariates),
            torch.ones(outcomes.size(0), device=outcomes.device)
        )

        loss_fls = self.loss_discriminator(
            self.discriminate(cf_outcomes_out, cf_treatments, covariates),
            torch.zeros(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
        )

        loss = (loss_tru+loss_fls)/2.
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()

    def update_eval_encoder(self):
        for target_param, param in zip(
            self.encoder_eval.parameters(), self.encoder.parameters()
        ):
            target_param.data.copy_(param.data)

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

    def init_outcome_emb(self):
        return MLP(
            [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
        )

    def init_treatment_emb(self):
        if self.type_treatments in ("object", "bool", "category", None):
            return CompoundEmbedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
        else:
            return MLP(
                [self.num_treatments] + [self.hparams["treatment_emb_dim"]] * 2
            )

    def init_covariates_emb(self):
        type_covariates = self.type_covariates
        if type_covariates is None or isinstance(type_covariates, str):
            type_covariates = [type_covariates] * len(self.num_covariates)

        covariates_emb = []
        for num_cov, type_cov in zip(self.num_covariates, type_covariates):
            if type_cov in ("object", "bool", "category", None):
                covariates_emb.append(CompoundEmbedding(
                        num_cov, self.hparams["covariate_emb_dim"]
                    ))
            else:
                covariates_emb.append(MLP(
                        [num_cov] + [self.hparams["covariate_emb_dim"]] * 2
                    ))
        return covariates_emb

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

    def move_input(self, input):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if isinstance(input, list):
            return [i.to(self.device) if i is not None else None for i in input]
        else:
            return input.to(self.device)

    def move_inputs(self, *inputs: torch.Tensor):
        """
        Move minibatch tensors to CPU/GPU.
        """
        return [self.move_input(i) if i is not None else None for i in inputs]

    def to_device(self, device):
        self.device = device
        self.to(self.device)

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for VCI
        """

        return self._set_hparams_(self, "")

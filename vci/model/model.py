import json

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .module import MLP, NegativeBinomial, ZeroInflatedNegativeBinomial

from ..utils.math_utils import (
    kldiv_normal,
    logprob_normal,
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
        outcome_dist=args["outcome_dist"],
        dist_mode=args["dist_mode"],
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

class VCI(torch.nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=False,
        embed_covariates=True,
        outcome_dist="normal",
        dist_mode="match",
        best_score=-1e3,
        patience=5,
        device="cuda",
        hparams=""
    ):
        super(VCI, self).__init__()
        # set generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.outcome_dist = outcome_dist
        self.dist_mode = dist_mode
        # early-stopping
        self.best_score = best_score
        self.patience = patience
        self.patience_trials = 0
        # distribution parameters
        if self.outcome_dist == "nb":
            self.num_dist_params = 2
        elif self.outcome_dist == "zinb":
            self.num_dist_params = 3
        elif self.outcome_dist == "normal":
            self.num_dist_params = 2
        else:
            raise ValueError("outcome_dist not recognized")

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
            "classifier_width": 64,
            "classifier_depth": 3,
            "discriminator_width": 64,
            "discriminator_depth": 3,
            "estimator_width": 64,
            "estimator_depth": 3,
            "indiv-spec_lh_weight": 1.0,
            "covar-spec_lh_weight": 1.7,
            "kl_divergence_weight": 0.1,
            "mc_sample_size": 30,
            "kde_kernel_std": 1.,
            "autoencoder_lr": 3e-4,
            "classifier_lr": 3e-4,
            "discriminator_lr": 3e-4,
            "estimator_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "classifier_wd": 4e-7,
            "discriminator_wd": 4e-7,
            "estimator_wd": 4e-7,
            "adversary_steps": 3,
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

        return self.hparams

    def _init_indiv_model(self):
        params = []

        # embeddings
        if self.embed_outcomes:
            self.outcomes_embeddings = MLP(
                [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
            )
            outcome_dim = self.hparams["outcome_emb_dim"]
            params.extend(list(self.outcomes_embeddings.parameters()))
        else:
            outcome_dim = self.num_outcomes

        if self.embed_treatments:
            self.treatments_embeddings = torch.nn.Embedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
            treatment_dim = self.hparams["treatment_emb_dim"]
            params.extend(list(self.treatments_embeddings.parameters()))
        else:
            treatment_dim = self.num_treatments

        if self.embed_covariates:
            self.covariates_embeddings = []
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, 
                        self.hparams["covariate_emb_dim"]
                    )
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )
            covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))
        else:
            covariate_dim = sum(self.num_covariates)

        # models
        self.encoder = MLP(
            [outcome_dim+treatment_dim+covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_dim"] * 2],
            final_act="relu"
        )
        params.extend(list(self.encoder.parameters()))

        self.decoder = MLP(
            [self.hparams["latent_dim"]+treatment_dim]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes * self.num_dist_params]
        )
        params.extend(list(self.decoder.parameters()))

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

        if self.dist_mode == "classify":
            self.treatment_classifier = MLP(
                [self.num_outcomes]
                + [self.hparams["classifier_width"]] * (self.hparams["classifier_depth"] - 1)
                + [self.num_treatments]
            )
            self.loss_treatment_classifier = torch.nn.CrossEntropyLoss()
            params = list(self.treatment_classifier.parameters())

            self.covariate_classifier = []
            self.loss_covariate_classifier = []
            for num_covariate in self.num_covariates:
                classifier = MLP(
                    [self.num_outcomes]
                    + [self.hparams["classifier_width"]]
                        * (self.hparams["classifier_depth"] - 1)
                    + [num_covariate]
                )
                self.covariate_classifier.append(classifier)
                self.loss_covariate_classifier.append(torch.nn.CrossEntropyLoss())
                params.extend(list(classifier.parameters()))

            self.optimizer_classifier = torch.optim.Adam(
                params,
                lr=self.hparams["classifier_lr"],
                weight_decay=self.hparams["classifier_wd"],
            )
            self.scheduler_classifier = torch.optim.lr_scheduler.StepLR(
                self.optimizer_classifier, step_size=self.hparams["step_size_lr"]
            )

            return self.treatment_classifier, self.covariate_classifier

        elif self.dist_mode == "discriminate":
            params = []

            # embeddings
            if self.embed_outcomes:
                self.adv_outcomes_emb = MLP(
                    [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
                )
                outcome_dim = self.hparams["outcome_emb_dim"]
                params.extend(list(self.adv_outcomes_emb.parameters()))
            else:
                outcome_dim = self.num_outcomes

            if self.embed_treatments:
                self.adv_treatments_emb = torch.nn.Embedding(
                    self.num_treatments, self.hparams["treatment_emb_dim"]
                )
                treatment_dim = self.hparams["treatment_emb_dim"]
                params.extend(list(self.adv_treatments_emb.parameters()))
            else:
                treatment_dim = self.num_treatments

            if self.embed_covariates:
                self.adv_covariates_emb = []
                for num_covariate in self.num_covariates:
                    self.adv_covariates_emb.append(
                        torch.nn.Embedding(num_covariate, 
                            self.hparams["covariate_emb_dim"]
                        )
                    )
                self.adv_covariates_emb = torch.nn.Sequential(
                    *self.adv_covariates_emb
                )
                covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))
            else:
                covariate_dim = sum(self.num_covariates)

            # model
            self.discriminator = MLP(
                [outcome_dim+treatment_dim+covariate_dim]
                + [self.hparams["discriminator_width"]] * (self.hparams["discriminator_depth"] - 1)
                + [1]
            )
            self.loss_discriminator = torch.nn.BCEWithLogitsLoss()
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
            params = []

            # embeddings
            if self.embed_treatments:
                self.adv_treatments_emb = torch.nn.Embedding(
                    self.num_treatments, self.hparams["treatment_emb_dim"]
                )
                treatment_dim = self.hparams["treatment_emb_dim"]
                params.extend(list(self.adv_treatments_emb.parameters()))
            else:
                treatment_dim = self.num_treatments

            if self.embed_covariates:
                self.adv_covariates_emb = []
                for num_covariate in self.num_covariates:
                    self.adv_covariates_emb.append(
                        torch.nn.Embedding(num_covariate, 
                            self.hparams["covariate_emb_dim"]
                        )
                    )
                self.adv_covariates_emb = torch.nn.Sequential(
                    *self.adv_covariates_emb
                )
                covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))
            else:
                covariate_dim = sum(self.num_covariates)

            # model
            self.outcome_estimator = MLP(
                [treatment_dim+covariate_dim]
                + [self.hparams["estimator_width"]] * (self.hparams["estimator_depth"] - 1)
                + [self.num_outcomes * self.num_dist_params]
            )
            self.loss_outcome_estimator = torch.nn.MSELoss()
            params.extend(list(self.outcome_estimator.parameters()))

            self.optimizer_estimator = torch.optim.Adam(
                params,
                lr=self.hparams["estimator_lr"],
                weight_decay=self.hparams["estimator_wd"],
            )
            self.scheduler_estimator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_estimator, step_size=self.hparams["step_size_lr"]
            )

            return self.outcome_estimator

        elif self.dist_mode == "match":
            return None

        else:
            raise ValueError("dist_mode not recognized")

    def encode(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covar.argmax(1)) 
                for covar, emb in zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        return self.encoder(inputs)

    def decode(self, latents, treatments):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))

        inputs = torch.cat([latents, treatments], -1)

        return self.decoder(inputs)

    def discriminate(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.adv_outcomes_emb(outcomes)
        if self.embed_treatments:
            treatments = self.adv_treatments_emb(treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covar.argmax(1)) 
                for covar, emb in zip(covariates, self.adv_covariates_emb)
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
            dist = self.outcome_dist

        if dist == "nb":
            mus = F.softplus(constructions[:, :dim]).add(eps)
            thetas = F.softplus(constructions[:, dim:]).add(eps)
            dist = NegativeBinomial(
                mu=mus, theta=thetas
            )
        elif dist == "zinb":
            mus = F.softplus(constructions[:, :dim]).add(eps)
            thetas = F.softplus(constructions[:, dim:(2*dim)]).add(eps)
            zi_logits = constructions[:, (2*dim):].add(eps)
            dist = ZeroInflatedNegativeBinomial(
                mu=mus, theta=thetas, zi_logits=zi_logits
            )
        elif dist == "normal":
            locs = constructions[:, :dim]
            scales = F.softplus(constructions[:, dim:]).add(eps)
            dist = Normal(
                loc=locs, scale=scales
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
                latents_dist.mean, latents_dist.stddev, treatments
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
            dist = self.outcome_dist

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

        return (logprob.sum(0)/num).mean()

    def loss(self, outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, covariates):
        """
        Compute losses.
        """
        # individual-specific likelihood
        indiv_spec_nllh = -outcomes_dist_samp.log_prob(
            outcomes.repeat(self.hparams["mc_sample_size"], 1)
        ).mean()

        # covariate-specific likelihood
        if self.dist_mode == "classify":
            raise NotImplementedError(
                'TODO: implement dist_mode "classify" for distribution loss')
        if self.dist_mode == "discriminate":
            if self.iteration % self.hparams["adversary_steps"]:
                self.update_discriminator(
                    outcomes, cf_outcomes_out.detach(), treatments, covariates
                )

            covar_spec_nllh = self.loss_discriminator(
                self.discriminate(cf_outcomes_out, treatments, covariates),
                torch.ones(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
            )
        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')
        elif self.dist_mode == "match":
            notNone = [o != None for o in cf_outcomes]
            cf_outcomes = [o for (o, n) in zip(cf_outcomes, notNone) if n]
            cf_outcomes_out = cf_outcomes_out[notNone]

            kernel_std = [self.hparams["kde_kernel_std"] * torch.ones_like(o) 
                for o in cf_outcomes]
            covar_spec_nllh = -self.logprob(
                cf_outcomes_out, (cf_outcomes, kernel_std), dist="normal"
            )

        # kl divergence
        kl_divergence = kldiv_normal(
            latents_dist.mean,
            latents_dist.stddev,
            cf_latents_dist.mean,
            cf_latents_dist.stddev
        )

        return (indiv_spec_nllh, covar_spec_nllh, kl_divergence)

    def update(self, outcomes, treatments, cf_outcomes, cf_treatments, covariates,
                sample=True, detach_pattern=None):
        """
        Update VCI's parameters given a minibatch of outcomes, treatments, and
        cell types.
        """
        outcomes, treatments, cf_outcomes, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_outcomes, cf_treatments, covariates
        )

        # q(z | y, x, t)
        latents_constr = self.encode(outcomes, treatments, covariates)
        latents_dist = self.distributionize(
            latents_constr, dim=self.hparams["latent_dim"], dist="normal"
        )

        # p(y | z, t)
        outcomes_constr_samp = self.sample(latents_dist.mean, latents_dist.stddev,
            treatments, size=self.hparams["mc_sample_size"]
        )
        outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        # p(y' | z, t')
        if sample:
            cf_outcomes_constr = self.decode(latents_dist.rsample(), cf_treatments)
            cf_outcomes_out = self.distributionize(cf_outcomes_constr).rsample()
        else:
            cf_outcomes_constr = self.decode(latents_dist.mean, cf_treatments)
            cf_outcomes_out = self.distributionize(cf_outcomes_constr).mean

        # q(z | y', x, t')
        if detach_pattern is None:
            cf_outcomes_in = cf_outcomes_out
        elif detach_pattern == "full":
            cf_outcomes_in = cf_outcomes_out.detach()
        elif detach_pattern == "half":
            if sample:
                cf_outcomes_in = self.distributionize(
                    self.decode(latents_dist.sample(), cf_treatments)
                ).rsample()
            else:
                cf_outcomes_in = self.distributionize(
                    self.decode(latents_dist.mean.detach(), cf_treatments)
                ).mean
        else:
            raise ValueError("Unrecognized: detaching pattern of the counterfactual outcome "
                "in the KL Divergence term.")
        cf_latents_constr = self.encode(
            cf_outcomes_in, cf_treatments, covariates
        )
        cf_latents_dist = self.distributionize(
            cf_latents_constr, dim=self.hparams["latent_dim"], dist="normal"
        )

        indiv_spec_nllh, covar_spec_nllh, kl_divergence = self.loss(
            outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, covariates
        )

        loss = (self.hparams["indiv-spec_lh_weight"] * indiv_spec_nllh
            + self.hparams["covar-spec_lh_weight"] * covar_spec_nllh
            + self.hparams["kl_divergence_weight"] * kl_divergence
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

    def update_discriminator(self, outcomes, cf_outcomes_out, treatments, covariates):
        loss_tru = self.loss_discriminator(
            self.discriminate(outcomes, treatments, covariates),
            torch.ones(outcomes.size(0), device=outcomes.device)
        )

        loss_fls = self.loss_discriminator(
            self.discriminate(cf_outcomes_out, treatments, covariates),
            torch.zeros(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
        )

        loss = (loss_tru+loss_fls)/2.
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()

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

import numpy as np

from sklearn.metrics import r2_score

import torch

from dpi.module import NegativeBinomial, ZeroInflatedNegativeBinomial

def evaluate_r2(autoencoder, dataset, dataset_control, min_samples=30):
    """
    Measures different quality metrics about an CPA `autoencoder`, when
    tasked to translate some `genes_control` into each of the perturbation/covariates
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.
    """

    mean_score, mean_score_de = [], []
    #var_score, var_score_de = [], []
    genes_control = dataset_control.genes
    perts_control = dataset_control.perturbations
    num = genes_control.size(0)

    for pert_category in np.unique(dataset.cov_pert):
        # pert_category category contains: 'cov_pert' info
        de_idx = np.where(
            dataset.var_names.isin(np.array(dataset.de_genes[pert_category]))
        )[0]

        idx = np.where(dataset.cov_pert == pert_category)[0]

        # estimate metrics only for reasonably-sized perturbation/cell-type combos
        if len(idx) > min_samples:
            perts = dataset.perturbations[idx][0].view(1, -1).repeat(num, 1).clone()
            covars = [
                covar[idx][0].view(1, -1).repeat(num, 1).clone()
                for covar in dataset.covariates
            ]

            genes_predict = autoencoder.predict(genes_control, perts_control, perts, covars)
            genes_predict = [gp.detach().cpu() for gp in genes_predict]

            if autoencoder.loss_ae == 'nb':
                dist = NegativeBinomial(
                    mu=genes_predict[0],
                    theta=genes_predict[1]
                )
                yp_m = dist.mean.mean(0)
                #yp_v = dist.variance.mean(0)
            elif autoencoder.loss_ae == 'zinb':
                dist = ZeroInflatedNegativeBinomial(
                    mu=genes_predict[0],
                    theta=genes_predict[1],
                    zi_logits=genes_predict[2]
                )
                yp_m = dist.mean.mean(0)
                #yp_v = dist.variance.mean(0)
            elif autoencoder.loss_ae == 'normal':
                yp_m = genes_predict[0].mean(0)
                #yp_v = genes_predict[1].mean(0)

            y_true = dataset.genes[idx, :].numpy()

            # true means and variances
            yt_m = y_true.mean(axis=0)
            #yt_v = y_true.var(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            #var_score.append(r2_score(yt_v, yp_v))

            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))
            #var_score_de.append(r2_score(yt_v[de_idx], yp_v[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [
            mean_score, mean_score_de#, var_score, var_score_de
        ]
    ]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    autoencoder.eval()
    with torch.no_grad():
        stats_test = evaluate_r2(
            autoencoder, 
            datasets["test"].subset_condition(control=False), 
            datasets["test"].subset_condition(control=True)
        )

        evaluation_stats = {
            "training": evaluate_r2(
                autoencoder,
                datasets["training"].subset_condition(control=False),
                datasets["training"].subset_condition(control=True)
            ),
            "test": stats_test,
            "ood": evaluate_r2(
                autoencoder,
                datasets["ood"],
                datasets["test"].subset_condition(control=True)
            ),
            "optimal for perturbations": 1 / datasets["test"].num_perturbations
            if datasets["test"].num_perturbations > 0
            else None,
        }
    autoencoder.train()
    return evaluation_stats

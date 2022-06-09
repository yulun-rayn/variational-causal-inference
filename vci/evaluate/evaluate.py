import numpy as np

from sklearn.metrics import r2_score

import torch

def evaluate_r2(model, dataset, dataset_control, min_samples=30):
    """
    Measures different quality metrics about an `model`, when
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

            yp = model.predict(genes_control, perts_control, perts, covars)
            yp_m = yp.mean(0).detach().cpu()

            # true means
            yt = dataset.genes[idx, :].numpy()
            yt_m = yt.mean(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]


def evaluate(model, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    model.eval()
    with torch.no_grad():
        stats_test = evaluate_r2(
            model, 
            datasets["test"].subset_condition(control=False), 
            datasets["test"].subset_condition(control=True)
        )

        evaluation_stats = {
            "training": evaluate_r2(
                model,
                datasets["training"].subset_condition(control=False),
                datasets["training"].subset_condition(control=True)
            ),
            "test": stats_test,
            "ood": evaluate_r2(
                model,
                datasets["ood"],
                datasets["test"].subset_condition(control=True)
            ),
            "optimal for perturbations": 1 / datasets["test"].num_perturbations
            if datasets["test"].num_perturbations > 0
            else None,
        }
    model.train()
    return evaluation_stats

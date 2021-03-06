import numpy as np

from sklearn.metrics import r2_score

import torch

from ..inference.inference import estimate

def evaluate(model, datasets, mode="r2", batch_size=None):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distribution (ood) splits.
    """

    model.eval()
    with torch.no_grad():
        if mode == "r2":
            evaluation_stats = {
                "training": evaluate_r2(
                    model,
                    datasets["training"].subset_condition(control=False),
                    datasets["training"].subset_condition(control=True),
                    batch_size=batch_size
                ),
                "test": evaluate_r2(
                    model, 
                    datasets["test"].subset_condition(control=False), 
                    datasets["test"].subset_condition(control=True),
                    batch_size=batch_size
                ),
                "ood": evaluate_r2(
                    model,
                    datasets["ood"],
                    datasets["test"].subset_condition(control=True),
                    batch_size=batch_size
                ),
                "optimal for perturbations": 1 / datasets["test"].num_perturbations
                if datasets["test"].num_perturbations > 0
                else None,
            }
        elif mode == "ci":
            evaluation_stats = {
                "training": evaluate_ci(
                    model, datasets["training"], batch_size=batch_size
                ),
                "test": evaluate_ci(
                    model, datasets["test"], batch_size=batch_size
                ),
                "ood": evaluate_ci(
                    model, datasets["ood"], batch_size=batch_size
                )
            }
        else:
            raise ValueError("mode not recognized")
    model.train()
    return evaluation_stats

def evaluate_r2(model, dataset, dataset_control, batch_size=None, min_samples=30):
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
    if batch_size is None:
        batch_size = num

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

            num_eval = 0
            yp = []
            while num_eval < num:
                end = min(num_eval+batch_size, num)
                out = model.predict(
                    genes_control[num_eval:end],
                    perts_control[num_eval:end],
                    perts[num_eval:end],
                    [covar[num_eval:end] for covar in covars]
                )
                yp.append(out.detach().cpu())

                num_eval += batch_size
            yp = torch.cat(yp, 0)
            yp_m = yp.mean(0)

            # true means
            yt = dataset.genes[idx, :].numpy()
            yt_m = yt.mean(axis=0)

            mean_score.append(r2_score(yt_m, yp_m))
            mean_score_de.append(r2_score(yt_m[de_idx], yp_m[de_idx]))

    return [
        np.mean(s) if len(s) else -1
        for s in [mean_score, mean_score_de]
    ]

def evaluate_ci(model, dataset, mode="ATT", batch_size=None):
    genes = dataset.genes.clone()
    perts = dataset.perturbations.clone()
    covars = [covar.clone() for covar in dataset.covariates]

    num = genes.size(0)
    if batch_size is None:
        batch_size = num

    names = dataset.pert_dose
    unique_names, indices, counts = np.unique(
        names, return_index=True, return_counts=True
    )
    target_perts = perts[indices]

    propensities = np.concatenate(
        [counts[np.nonzero(unique_names==n)[0]] for n in names]
    )

    estimates = []
    for n, p in zip(unique_names, target_perts):
        cf_perts = torch.tensor(p).repeat(perts.size(0), 1)

        num_eval = 0
        predicts = []
        while num_eval < num:
            end = min(num_eval+batch_size, num)
            out = model.predict(
                genes[num_eval:end],
                perts[num_eval:end],
                cf_perts[num_eval:end],
                [covar[num_eval:end] for covar in covars]
            )
            predicts.append(out.detach().cpu())

            num_eval += batch_size
        predicts = torch.cat(predicts, 0)

        estimates.append(
            estimate(mode,
                outcomes=genes.numpy(), treatments=names,
                predicts=predicts.numpy(), propensities=propensities,
                target_treatment=n
            )
        )

    return dict(zip(unique_names, estimates))

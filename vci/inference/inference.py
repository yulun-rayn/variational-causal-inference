import torch

def estimate(type='ATT', *args, **kwargs):
    if type=='ATT':
        mean_est, std_est = estimate_ATT(*args, **kwargs, return_all=False)
    elif type=='ATE':
        mean_est, std_est = estimate_ATE(*args, **kwargs, return_all=False)
    else:
        raise ValueError("type not recognized")

    return {
        "type": type,
        "mean": mean_est.tolist(),
        "std": std_est.tolist()
    }

def estimate_ATE(outcomes, treatments, predicts, propensities,
                 treatment, control, return_all=False):
    treatment_effects = estimate_ATT(
        outcomes, treatments, predicts, propensities,
        treatment, return_all=True
    )
    control_effects = estimate_ATT(
        outcomes, treatments, predicts, propensities,
        control, return_all=True
    )

    estimates = treatment_effects - control_effects

    if return_all:
        return estimates
    return estimates.mean(0), estimates.std(0)

def estimate_ATT(outcomes, treatments, predicts, propensities,
                 target_treatment, return_all=False):
    estimates = (
        ((treatments==target_treatment)/propensities)[:, None] 
            * (outcomes - predicts) 
        + predicts
    )

    if return_all:
        return estimates
    return estimates.mean(0), estimates.std(0)

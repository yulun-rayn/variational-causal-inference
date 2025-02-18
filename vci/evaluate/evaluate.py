from collections import defaultdict

from .gene_evaluate import gene_evaluate
from .image_evaluate import image_evaluate

from ..utils.data_utils import move_tensors

def evaluate(model, datasets, data_name="image", **kwargs):
    if data_name == "gene":
        return gene_evaluate(model, datasets, **kwargs)
    elif data_name in ("celebA", "morphoMNIST"):
        return image_evaluate(model, datasets, **kwargs)
    else:
        raise ValueError("name not recognized")

def evaluate_loss(model, datasets, **kwargs):
    model.eval()
    epoch_eval_stats = defaultdict(float)
    for batch_idx, batch in enumerate(datasets["test_loader"]):
        minibatch_eval_stats = model.evaluate(
            move_tensors(*batch, device=model.device), batch_idx
        )

        for key, val in minibatch_eval_stats.items():
            epoch_eval_stats[key] += val

    for key, val in epoch_eval_stats.items():
        epoch_eval_stats[key] = val / len(datasets["test_loader"])
    model.train()
    return epoch_eval_stats, model.early_stopping(None)

import os
import copy

import numpy as np

import torch
from torchvision.utils import save_image

from ..utils.data_utils import move_tensors

def image_evaluate(model, datasets, save_dir="", epoch=-1, sample_size=3, save_orig=True, **kwargs):
    dataset = datasets["test"]
    labels_eval = [np.unique(dataset.labels[:, i].tolist()) for i in range(dataset.labels.shape[1])]

    model.eval()
    with torch.no_grad():
        for idx in np.random.choice(len(dataset), sample_size, replace=False):
            data, _, covars, _, _ = dataset[idx]
            label = np.array(dataset.labels[idx])
            for i, labs_e in enumerate(labels_eval):
                if len(labs_e) > 100: # if too many labels to traverse
                    labs_e = labs_e[np.arange(0, len(labs_e), int(len(labs_e)/100))]
                dat = data.repeat(len(labs_e), *[1]*data.dim())
                lab = np.tile(label, (len(labs_e), 1))
                covs = [cov.repeat(len(labs_e), 1) for cov in covars]
                cf_lab = copy.deepcopy(lab)
                cf_lab[:, i] = labs_e

                if dataset.target_transform:
                    lab = dataset.target_transform(lab)
                    cf_lab = dataset.target_transform(cf_lab)

                outs = model.predict(*move_tensors(
                    dat, lab, covs, cf_lab, device=model.device
                )).detach().cpu()

                if save_orig:
                    outs = torch.cat([data.unsqueeze(0), outs], dim=0)
                save_image(outs,
                    os.path.join(save_dir,
                        'epoch-' + str(epoch) + '_sample-' + str(idx) + '_label-' + str(i) + '.png'
                    ),
                    normalize=False, scale_each=True
                )
    model.train()
    return [], model.early_stopping(None)

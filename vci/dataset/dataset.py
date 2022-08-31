import sys
from typing import Union

import scipy
import numpy as np
import scanpy as sc

import torch

from ..utils.general_utils import unique_ind
from ..utils.data_utils import check_adata, rank_genes_groups

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


class Dataset:
    def __init__(
        self,
        data,
        perturbation_key="perturbation",
        control_key="control",
        dose_key=None,
        covariate_keys=None,
        split_key=None,
        test_ratio=0.2,
        random_state=42,
        sample_cf=False,
        cf_samples=20
    ):
        if type(data) == str:
            data = sc.read(data)

        self.sample_cf = sample_cf
        self.cf_samples = cf_samples

        # Fields
        # perturbation
        if "perturbation" in data.uns["fields"]:
            perturbation_key = data.uns["fields"]["perturbation"]
        else:
            assert perturbation_key in data.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"
        # control
        if "control" in data.uns["fields"]:
            control_key = data.uns["fields"]["control"]
        else:
            assert control_key in data.obs.columns, f"Control {control_key} is missing in the provided adata"
        # dose
        if "dose" in data.uns["fields"]:
            dose_key = data.uns["fields"]["dose"]
        elif dose_key is None:
            print("Adding a dummy dose...")
            data.obs["dummy_dose"] = 1.0
            dose_key = "dummy_dose"
        else:
            assert dose_key in data.obs.columns, f"Dose {dose_key} is missing in the provided adata"
        # covariates
        if "covariates" in data.uns["fields"]:
            covariate_keys = list(data.uns["fields"]["covariates"])
        elif covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            data.obs["dummy_covar"] = "dummy_covar"
            covariate_keys = ["dummy_covar"]
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in data.obs.columns, f"Covariate {key} is missing in the provided adata"
        # split
        if "split" in data.uns["fields"]:
            split_key = data.uns["fields"]["split"]
        elif split_key is None:
            print(f"Performing automatic train-test split with {test_ratio} ratio.")
            from sklearn.model_selection import train_test_split

            data.obs["split"] = "train"
            idx_train, idx_test = train_test_split(
                data.obs_names, test_size=test_ratio, random_state=random_state
            )
            data.obs["split"].loc[idx_train] = "train"
            data.obs["split"].loc[idx_test] = "test"
            split_key = "split"
        else:
            assert split_key in data.obs.columns, f"Split {split_key} is missing in the provided adata"

        self.indices = {
            "all": list(range(len(data.obs))),
            "control": np.where(data.obs[control_key] == 1)[0].tolist(),
            "treated": np.where(data.obs[control_key] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

        self.perturbation_key = perturbation_key
        self.control_key = control_key
        self.dose_key = dose_key
        self.covariate_keys = covariate_keys

        self.control_names = np.unique(
            data[data.obs[self.control_key] == 1].obs[self.perturbation_key]
        )

        if scipy.sparse.issparse(data.X):
            self.genes = torch.Tensor(data.X.A)
        else:
            self.genes = torch.Tensor(data.X) # data.layers["counts"]

        self.var_names = data.var_names

        data, replaced = check_adata(
            data, [perturbation_key, dose_key] + covariate_keys
        )

        self.pert_names = np.array(data.obs[perturbation_key].values)
        self.doses = np.array(data.obs[dose_key].values)

        # get unique perturbations
        pert_unique = np.array(self.get_unique_perts())

        # store as attribute for molecular featurisation
        pert_unique_onehot = torch.eye(len(pert_unique))

        self.perts_dict = dict(
            zip(pert_unique, pert_unique_onehot)
        )

        # get perturbation combinations
        perturbations = []
        for i, comb in enumerate(self.pert_names):
            perturbation_combos = [self.perts_dict[p] for p in comb.split("+")]
            dose_combos = str(data.obs[dose_key].values[i]).split("+")
            perturbation_ohe = []
            for j, d in enumerate(dose_combos):
                perturbation_ohe.append(float(d) * perturbation_combos[j])
            perturbations.append(sum(perturbation_ohe))
        self.perturbations = torch.stack(perturbations)

        self.controls = data.obs[self.control_key].values.astype(bool)

        if covariate_keys is not None:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            cov_names = []
            self.covars_dict = {}
            self.covariates = []
            self.num_covariates = []
            for cov in covariate_keys:
                values = np.array(data.obs[cov].values)
                cov_names.append(values)

                names = np.unique(values)
                self.num_covariates.append(len(names))

                names_onehot = torch.eye(len(names))
                self.covars_dict[cov] = dict(
                    zip(list(names), names_onehot)
                )

                self.covariates.append(
                    torch.stack([self.covars_dict[cov][v] for v in values])
                )
            self.cov_names = np.array(["_".join(c) for c in zip(*cov_names)])
        else:
            self.cov_names = np.array([""] * len(data))
            self.covars_dict = None
            self.covariates = None
            self.num_covariates = None

        self.num_genes = self.genes.shape[1]
        self.num_perturbations = len(pert_unique)

        self.cov_pert = np.array([
            f"{self.cov_names[i]}_"
            f"{data.obs[perturbation_key].values[i]}"
            for i in range(len(data))
        ])
        self.pert_dose = np.array([
            f"{data.obs[perturbation_key].values[i]}"
            f"_{data.obs[dose_key].values[i]}"
            for i in range(len(data))
        ])
        self.cov_pert_dose = np.array([
            f"{self.cov_names[i]}_{self.pert_dose[i]}"
            for i in range(len(data))
        ])

        if not ("rank_genes_groups_cov" in data.uns) or replaced:
            data.obs["cov_name"] = self.cov_names
            data.obs["cov_pert_name"] = self.cov_pert
            print("Ranking genes for DE genes.")
            rank_genes_groups(data,
                groupby="cov_pert_name", 
                reference="cov_name",
                control_key=control_key)
        self.de_genes = data.uns["rank_genes_groups_cov"]

    def get_unique_perts(self, all_perts=None):
        if all_perts is None:
            all_perts = self.pert_names
        perts = [i for p in all_perts for i in p.split("+")]
        return list(dict.fromkeys(perts))

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.sample_cf = dataset.sample_cf
        self.cf_samples = dataset.cf_samples

        self.perturbation_key = dataset.perturbation_key
        self.control_key = dataset.control_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys

        self.control_names = dataset.control_names

        self.perts_dict = dataset.perts_dict
        self.covars_dict = dataset.covars_dict

        self.genes = dataset.genes[indices]
        self.perturbations = indx(dataset.perturbations, indices)
        self.controls = dataset.controls[indices]
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.pert_names = indx(dataset.pert_names, indices)
        self.doses = indx(dataset.doses, indices)

        self.cov_names = indx(dataset.cov_names, indices)
        self.cov_pert = indx(dataset.cov_pert, indices)
        self.pert_dose = indx(dataset.pert_dose, indices)
        self.cov_pert_dose = indx(dataset.cov_pert_dose, indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_perturbations = dataset.num_perturbations

        self.cov_pert_dose_idx = unique_ind(self.cov_pert_dose)

    def subset_condition(self, control=True):
        if control is None:
            return self
        else:
            idx = np.where(self.controls == control)[0].tolist()
            return SubDataset(self, idx)

    def __getitem__(self, i):
        cf_pert_dose_name = self.control_names[0]
        while any(c in cf_pert_dose_name for c in self.control_names):
            cf_i = np.random.choice(len(self.pert_dose))
            cf_pert_dose_name = self.pert_dose[cf_i]

        cf_genes = None
        if self.sample_cf:
            covariate_name = indx(self.cov_names, i)
            cf_name = covariate_name + f"_{cf_pert_dose_name}"

            if cf_name in self.cov_pert_dose_idx:
                cf_inds = self.cov_pert_dose_idx[cf_name]
                cf_genes = self.genes[np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))]

        return (
            self.genes[i],
            indx(self.perturbations, i),
            cf_genes,
            indx(self.perturbations, cf_i),
            *[indx(cov, i) for cov in self.covariates]
        )

    def __len__(self):
        return len(self.genes)

def load_dataset_splits(
    data_path: str,
    perturbation_key: str = "perturbation",
    control_key: str = "control",
    dose_key: str = None,
    covariate_keys: Union[list, str] = None,
    split_key: str = None,
    sample_cf: bool = False,
    return_dataset: bool = False,
):

    dataset = Dataset(
        data_path, perturbation_key, control_key, dose_key, covariate_keys, split_key, 
        sample_cf=sample_cf
    )

    splits = {
        "training": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("ood", "all"),
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits

indx = lambda a, i: a[i] if a is not None else None

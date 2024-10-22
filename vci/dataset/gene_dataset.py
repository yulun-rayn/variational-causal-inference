import sys

import scipy
import numpy as np
import pandas as pd
import scanpy as sc

import torch

from ..utils.general_utils import unique_ind

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


class GeneDataset:
    def __init__(
        self,
        data,
        perturbation_key="perturbation",
        control_key="control",
        dose_key="dose",
        covariate_keys="covariates",
        split_key="split",
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
        if perturbation_key in data.uns["fields"]:
            perturbation_key = data.uns["fields"][perturbation_key]
        else:
            assert perturbation_key in data.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"
        # control
        if control_key in data.uns["fields"]:
            control_key = data.uns["fields"][control_key]
        else:
            assert control_key in data.obs.columns, f"Control {control_key} is missing in the provided adata"
        # dose
        if dose_key in data.uns["fields"]:
            dose_key = data.uns["fields"][dose_key]
        elif dose_key is None:
            print("Adding a dummy dose...")
            data.obs["dummy_dose"] = 1.0
            dose_key = "dummy_dose"
        else:
            assert dose_key in data.obs.columns, f"Dose {dose_key} is missing in the provided adata"
        # covariates
        if isinstance(covariate_keys, str) and covariate_keys in data.uns["fields"]:
            covariate_keys = list(data.uns["fields"][covariate_keys])
        elif covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            data.obs["dummy_covar"] = "dummy-covar"
            covariate_keys = ["dummy_covar"]
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in data.obs.columns, f"Covariate {key} is missing in the provided adata"
        # split
        if split_key in data.uns["fields"]:
            split_key = data.uns["fields"][split_key]
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

        self.num_outcomes = self.genes.shape[1]
        self.num_treatments = len(pert_unique)

        self.nb_dims = self.genes[0].ndim

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

        if not ("rank_genes_groups_cov" in data.uns):
            data.obs["cov_name"] = self.cov_names
            data.obs["cov_pert_name"] = self.cov_pert
            print("Ranking genes for DE genes...")
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
        return GeneSubDataset(self, idx)

    def __len__(self):
        return len(self.genes)


class GeneSubDataset:
    """
    Subsets a `GeneDataset` by selecting the examples given by `indices`.
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
        self.perturbations = self.indx(dataset.perturbations, indices)
        self.controls = dataset.controls[indices]
        self.covariates = [self.indx(cov, indices) for cov in dataset.covariates]

        self.pert_names = self.indx(dataset.pert_names, indices)
        self.doses = self.indx(dataset.doses, indices)

        self.cov_names = self.indx(dataset.cov_names, indices)
        self.cov_pert = self.indx(dataset.cov_pert, indices)
        self.pert_dose = self.indx(dataset.pert_dose, indices)
        self.cov_pert_dose = self.indx(dataset.cov_pert_dose, indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_outcomes = dataset.num_outcomes
        self.num_treatments = dataset.num_treatments

        self.nb_dims = dataset.nb_dims

        if self.sample_cf:
            self.cov_pert_dose_idx = unique_ind(self.cov_pert_dose)

    def indx(self, a, i):
        return a[i] if a is not None else None

    def subset_condition(self, control=True):
        if control is None:
            return self
        else:
            idx = np.where(self.controls == control)[0].tolist()
            return GeneSubDataset(self, idx)

    def __getitem__(self, i):
        cf_pert_dose_name = self.control_names[0]
        while any(c in cf_pert_dose_name for c in self.control_names):
            cf_i = np.random.choice(len(self.pert_dose))
            cf_pert_dose_name = self.pert_dose[cf_i]

        cf_genes = None
        if self.sample_cf:
            covariate_name = self.indx(self.cov_names, i)
            cf_name = covariate_name + f"_{cf_pert_dose_name}"

            if cf_name in self.cov_pert_dose_idx:
                cf_inds = self.cov_pert_dose_idx[cf_name]
                cf_genes = self.genes[np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))]

        return ( # Y, T, X, T', Ys under X and T'
            self.genes[i],
            self.indx(self.perturbations, i),
            [self.indx(cov, i) for cov in self.covariates],
            self.indx(self.perturbations, cf_i),
            cf_genes
        )

    def __len__(self):
        return len(self.genes)

def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        print(cov_cat)
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

def rank_genes_groups(
    adata,
    groupby,
    reference,
    control_key,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    for cov_cat in np.unique(adata.obs[reference].values):
        adata_cov = adata[adata.obs[reference] == cov_cat]
        control_group_cov = (
            adata_cov[adata_cov.obs[control_key] == 1].obs[groupby].values[0]
        )

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            method='t-test' # TODO(Y): remove this for future version of scanpy
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

def ranks_to_df(data, key="rank_genes_groups"):
    """Converts an `sc.tl.rank_genes_groups` result into a MultiIndex dataframe.

    You can access various levels of the MultiIndex with `df.loc[[category]]`.

    Params
    ------
    data : `AnnData`
    key : str (default: 'rank_genes_groups')
        Field in `.uns` of data where `sc.tl.rank_genes_groups` result is
        stored.
    """
    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == "params":
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)

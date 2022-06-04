import re
import sys
import random
import collections
from typing import Union

import scipy
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.preprocessing import OneHotEncoder

import torch
from torch._six import string_classes

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)


class Dataset:
    def __init__(
        self,
        data,
        perturbation_key,
        control_key,
        dose_key=None,
        covariate_keys=None,
        split_key=None,
        test_ratio=0.2,
        random_state=42,
        dist_mode='match',
        cf_samples=30
    ):
        if type(data) == str:
            data = sc.read(data)
        # Assert perturbation and control keys are present in the adata object
        assert perturbation_key in data.obs.columns, f"Perturbation {perturbation_key} is missing in the provided adata"
        assert control_key in data.obs.columns, f"Perturbation {control_key} is missing in the provided adata"

        self.dist_mode = dist_mode
        self.cf_samples = cf_samples

        # If no covariates, create dummy covariate
        if covariate_keys is None or len(covariate_keys)==0:
            print("Adding a dummy covariate...")
            data.obs['dummy_covar'] = 'dummy_covar'
            covariate_keys = ['dummy_covar']
        else:
            if not isinstance(covariate_keys, list):
                covariate_keys = [covariate_keys]
            for key in covariate_keys:
                assert key in data.obs.columns, f"Covariate {key} is missing in the provided adata"
        # If no dose, create dummy dose
        if dose_key is None:
            print("Adding a dummy dose...")
            data.obs['dummy_dose'] = 1.0
            dose_key = 'dummy_dose'
        else:
            assert dist_mode != 'match', f"Dose {dose_key} cannot be handled with dist_mode 'match'"
            assert dose_key in data.obs.columns, f"Dose {dose_key} is missing in the provided adata"
        # If no split, create split
        if split_key is None:
            print(f"Performing automatic train-test split with {test_ratio} ratio.")
            from sklearn.model_selection import train_test_split

            data.obs['split'] = "train"
            idx_train, idx_test = train_test_split(
                data.obs_names, test_size=test_ratio, random_state=random_state
            )
            data.obs['split'].loc[idx_train] = "train"
            data.obs['split'].loc[idx_test] = "test"
            split_key = 'split'
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

        self.control_name = list(
            np.unique(data[data.obs[self.control_key] == 1].obs[self.perturbation_key])
        ) # TODO(Y): remove this in the future

        if scipy.sparse.issparse(data.X):
            self.genes = torch.Tensor(data.X.A)
        else:
            self.genes = torch.Tensor(data.X) # data.layers['counts']

        self.var_names = data.var_names

        data, replaced = check_adata(
            data, [perturbation_key, dose_key] + covariate_keys
        )

        if not ("cov_pert_name" in data.obs) or replaced:
            print("Creating 'cov_pert_name' field.")
            cov_pert_name = []
            for i in range(len(data)):
                comb_name = ""
                for cov_key in self.covariate_keys:
                    comb_name += f"{data.obs[cov_key].values[i]}_"
                comb_name += f"{data.obs[perturbation_key].values[i]}"
                cov_pert_name.append(comb_name)
            data.obs["cov_pert_name"] = cov_pert_name

        if not ("rank_genes_groups_cov" in data.uns) or replaced:
            print("Ranking genes for DE genes.")
            rank_genes_groups(data, groupby="cov_pert_name", control_key=control_key)

        self.cov_pert = np.array(data.obs["cov_pert_name"].values)
        self.de_genes = data.uns["rank_genes_groups_cov"]

        self.pert_names = np.array(data.obs[perturbation_key].values)
        self.doses = np.array(data.obs[dose_key].values)

        # get unique perturbations
        pert_unique = set()
        for d in self.pert_names:
            [pert_unique.add(i) for i in d.split("+")]
        pert_unique = np.array(list(pert_unique))

        # encode perturbations
        encoder_pert = OneHotEncoder(sparse=False)
        encoder_pert.fit(pert_unique.reshape(-1, 1))

        # store as attribute for molecular featurisation
        pert_unique_onehot = encoder_pert.transform(pert_unique.reshape(-1, 1))

        self.perts_dict = dict(
            zip(
                pert_unique,
                torch.Tensor(pert_unique_onehot),
            )
        )

        # get perturbation combinations
        perturbations = []
        for i, comb in enumerate(self.pert_names):
            perturbation_combos = encoder_pert.transform(
                np.array(comb.split("+")).reshape(-1, 1)
            )
            dose_combos = str(data.obs[dose_key].values[i]).split("+")
            perturbation_ohe = []
            for j, d in enumerate(dose_combos):
                perturbation_ohe.append(float(d) * perturbation_combos[j])
            perturbations.append(sum(perturbation_ohe))
        self.perturbations = torch.Tensor(perturbations)

        self.controls = data.obs[self.control_key].values.astype(bool)

        if covariate_keys is not None:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covars_dict = {}
            self.covariates = []
            self.num_covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)

                names = np.unique(self.covariate_names[cov])
                self.num_covariates.append(len(names))

                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                names_onehot = encoder_cov.transform(names.reshape(-1, 1))
                self.covars_dict[cov] = dict(
                    zip(list(names), torch.Tensor(names_onehot))
                )

                names = self.covariate_names[cov]
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1)))
                )
        else:
            self.covariate_names = None
            self.covars_dict = None
            self.covariates = None
            self.num_covariates = None

        self.num_genes = self.genes.shape[1]
        self.num_perturbations = len(pert_unique)

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __getitem__(self, i):
        if self.dist_mode == 'classify':
            cf_genes = None
        elif self.dist_mode == 'estimate':
            cf_genes = None
        elif self.dist_mode == 'match':
            covariate_name = [indx(self.covariate_names[cov], i) for cov in list(self.covariate_names)]

            cf_pert_name = self.control_name
            while cf_pert_name == self.control_name:
                cf_pert_name = np.random.choice(self.pert_names)
            cf_name = '_'.join(covariate_name) + f"_{cf_pert_name}"
            cf_inds = np.nonzero(self.cov_pert==cf_name)[0]

            cf_genes = self.genes[np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))]

        return (
                self.genes[i],
                indx(self.perturbations, i),
                cf_genes,
                self.perts_dict[cf_pert_name],
                *[indx(cov, i) for cov in self.covariates]
            )

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.dist_mode = dataset.dist_mode
        self.cf_samples = dataset.cf_samples

        self.perturbation_key = dataset.perturbation_key
        self.control_key = dataset.control_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys

        self.control_name = indx(dataset.control_name, 0) #TODO(Y): remove this in the future

        self.perts_dict = dataset.perts_dict
        self.covars_dict = dataset.covars_dict

        self.genes = dataset.genes[indices]
        self.perturbations = indx(dataset.perturbations, indices)
        self.controls = dataset.controls[indices]
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.pert_names = indx(dataset.pert_names, indices)
        self.doses = indx(dataset.doses, indices)

        self.cov_pert = indx(dataset.cov_pert, indices)
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_perturbations = dataset.num_perturbations

    def subset_condition(self, control=True):
        idx = np.where(self.controls == control)[0].tolist()
        return SubDataset(self, idx)

    def __getitem__(self, i):
        if self.dist_mode == 'classify':
            cf_genes = None
        elif self.dist_mode == 'estimate':
            cf_genes = None
        elif self.dist_mode == 'match':
            covariate_name = [indx(self.covariate_names[cov], i) for cov in list(self.covariate_names)]

            cf_pert_name = self.control_name
            while cf_pert_name == self.control_name:
                cf_pert_name = np.random.choice(self.pert_names)
            cf_name = '_'.join(covariate_name) + f"_{cf_pert_name}"
            cf_inds = np.nonzero(self.cov_pert==cf_name)[0]

            cf_genes = self.genes[np.random.choice(cf_inds, min(len(cf_inds), self.cf_samples))]

        return (
                self.genes[i],
                indx(self.perturbations, i),
                cf_genes,
                self.perts_dict[cf_pert_name],
                *[indx(cov, i) for cov in self.covariates]
            )

    def __len__(self):
        return len(self.genes)

def load_dataset_splits(
    data: str,
    perturbation_key: Union[str, None],
    control_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    split_key: str,
    dist_mode: str,
    return_dataset: bool = False,
):

    dataset = Dataset(
        data, perturbation_key, control_key, dose_key, covariate_keys, split_key, dist_mode=dist_mode
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

    covars_comb = []
    for i in range(len(adata)):
        cov = "_".join(adata.obs["cov_pert_name"].values[i].split("_")[:-1])
        covars_comb.append(cov)
    adata.obs["covars_comb"] = covars_comb

    gene_dict = {}
    for cov_cat in np.unique(adata.obs["covars_comb"].values):
        adata_cov = adata[adata.obs["covars_comb"] == cov_cat]
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


def check_adata(adata, special_fields, special_chars=["_"], replacements=["-"]):
    replaced = False
    for sf in special_fields:
        if sf is None:
            continue
        chars, replaces = [], []
        for i, el in enumerate(adata.obs[sf].values):
            for char, replace in zip(special_chars, replacements):
                if char in str(el):
                    adata.obs[sf][i] = [s.replace(char, replace) for s in adata.obs[sf].values]

                    if char not in chars:
                        chars.append(char)
                        replaces.append(replace)
                    replaced = True
        if len(chars) > 0:
            print(
                f"WARNING. Special characters {chars} were found in: '{sf}'.",
                f"They will be replaced with {replaces}.",
                "Be careful, it may lead to errors downstream.",
            )

    return adata, replaced

np_str_obj_array_pattern = re.compile(r'[SaUO]')

data_collate_err_msg_format = (
    "data_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def data_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return list(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(data_collate_err_msg_format.format(elem.dtype))

            return data_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: data_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: data_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(data_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [data_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([data_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [data_collate(samples) for samples in transposed]

    raise TypeError(data_collate_err_msg_format.format(elem_type))

indx = lambda a, i: a[i] if a is not None else None

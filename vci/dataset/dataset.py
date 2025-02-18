import sys

from .gene_dataset import GeneDataset
from .image_dataset import ImageDataset
from .celebA_dataset import CelebADataset
from .morphoMNIST_dataset import MorphoMNISTDataset

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

def load_dataset_splits(data_name, data_path, label_names=None, sample_cf=False):
    if data_name == "gene":
        dataset = GeneDataset(
            data_path, "perturbation", "control", "dose", "covariates", "split", 
            sample_cf=sample_cf
        )

        return {
            "train": dataset.subset("train", "all"),
            "test": dataset.subset("test", "all"),
            "ood": dataset.subset("ood", "all"),
        }
    elif data_name == "celebA":
        if label_names is None:
            label_names = [15, 31]

        return {
            "train": CelebADataset(data_path, label_idx=label_names, split="train"),
            "valid": CelebADataset(data_path, label_idx=label_names, split="valid"),
            "test": CelebADataset(data_path, label_idx=label_names, split="test"),
        }
    elif data_name == "morphoMNIST":
        if label_names is None:
            label_names = ["thickness", "intensity"]
        dataset = MorphoMNISTDataset(data_path, label_names=label_names)

        return {
            "train": dataset.get_split("train"),
            "test": dataset.get_split("test"),
        }
    else:
        raise ValueError("data_name not recognized")

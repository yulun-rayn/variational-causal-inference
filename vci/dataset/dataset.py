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

def load_dataset_splits(
    data_name: str,
    data_path: str,
    sample_cf: bool = False
):

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
        return {
            "train": CelebADataset(data_path, split="train"),
            "valid": CelebADataset(data_path, split="valid"),
            "test": CelebADataset(data_path, split="test"),
        }
    elif data_name == "morphoMNIST":
        dataset = MorphoMNISTDataset(data_path)

        return {
            "train": dataset.get_split("train"),
            "test": dataset.get_split("test"),
        }
    else:
        raise ValueError("data_name not recognized")

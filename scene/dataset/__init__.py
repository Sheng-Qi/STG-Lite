from importlib import import_module
from typing import Type
from scene.dataset.abstract_dataset import AbstractDataset


def parse_dataset(dataset_type: str) -> Type[AbstractDataset]:
    DATASET_DICT = {
        "technicolor": ["technicolor_dataset", "TechnicolorDataset"],
    }
    dataset_module = import_module("scene.dataset." + DATASET_DICT[dataset_type][0])
    return getattr(dataset_module, DATASET_DICT[dataset_type][1])

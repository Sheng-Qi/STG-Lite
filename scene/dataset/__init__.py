from importlib import import_module
from typing import Type, Literal
from scene.dataset.abstract_dataset import AbstractDataset

DatasetNames = Literal["technicolor", "basic_colmap", "movingrig"]


def parse_dataset(dataset_type: DatasetNames) -> Type[AbstractDataset]:
    DATASET_DICT = {
        "technicolor": ["technicolor_dataset", "TechnicolorDataset"],
        "basic_colmap": ["basic_colmap_dataset", "BasicColmapDataset"],
        "movingrig": ["movingrig_dataset", "MovingRigDataset"],
    }
    dataset_module = import_module("scene.dataset." + DATASET_DICT[dataset_type][0])
    return getattr(dataset_module, DATASET_DICT[dataset_type][1])

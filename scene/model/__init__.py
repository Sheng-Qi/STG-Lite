from importlib import import_module
from typing import Type
from scene.model.abstract_gaussian_model import AbstractModel


def parse_model(model: str) -> Type[AbstractModel]:
    MODEL_DICT = {
        "basic": ["basic_gaussian_model", "BasicGaussianModel"],
        "spacetime": ["spacetime_gaussian_model", "SpacetimeGaussianModel"],
    }
    model_module = import_module("scene.model." + MODEL_DICT[model][0])
    return getattr(model_module, MODEL_DICT[model][1])

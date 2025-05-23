from importlib import import_module
from typing import Type, Literal
from scene.model.abstract_gaussian_model import AbstractModel

ModelNames = Literal["basic", "spacetime", "spacetime_360", "stree"]


def parse_model(model: ModelNames) -> Type[AbstractModel]:
    MODEL_DICT = {
        "basic": ["basic_gaussian_model", "BasicGaussianModel"],
        "spacetime": ["spacetime_gaussian_model", "SpacetimeGaussianModel"],
        "spacetime_360": ["spacetime_360", "Spacetime360Model"],
        "stree": ["stree_model", "STreeModel"],
    }
    model_module = import_module("scene.model." + MODEL_DICT[model][0])
    return getattr(model_module, MODEL_DICT[model][1])


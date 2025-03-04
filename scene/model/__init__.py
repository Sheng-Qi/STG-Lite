from importlib import import_module
from typing import Type, Literal
from scene.model.abstract_gaussian_model import AbstractModel

ModelNames = Literal["basic", "spacetime", "spacetime_360"]


def parse_model(model: ModelNames) -> Type[AbstractModel]:
    MODEL_DICT = {
        "basic": ["basic_gaussian_model", "BasicGaussianModel"],
        "spacetime": ["spacetime_gaussian_model", "SpacetimeGaussianModel"],
        "spacetime_360": ["spacetime_360", "Spacetime360Model"],
    }
    model_module = import_module("scene.model." + MODEL_DICT[model][0])
    return getattr(model_module, MODEL_DICT[model][1])


def parse_cfg_args(model: ModelNames) -> str:
    MODEL_DICT = {
        "basic": "Namespace(sh_degree=0)",
        "spacetime": "Namespace(sh_degree=3)",
        "spacetime_360": "Namespace(sh_degree=3)",
    }
    return MODEL_DICT[model]

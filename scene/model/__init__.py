from importlib import import_module
from typing import Type
from scene.model.abstract_gaussian_model import AbstractModel


def parse_model(model: str) -> Type[AbstractModel]:
    model_module = import_module(f"scene.model.{model}")
    return getattr(model_module, "GaussianModel")

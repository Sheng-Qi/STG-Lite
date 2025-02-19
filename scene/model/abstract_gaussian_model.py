from typing import Type
import torch
import logging
from abc import abstractmethod
from utils.graphics_utils import BasicPointCloud
from scene.dataset.abstract_dataset import AbstractDataset
from scene.cameras import Camera


class AbstractModel:
    @abstractmethod
    def __init__(self, model_params: dict):
        try:
            self._device = torch.device(model_params["device"])
        except Exception as e:
            logging.error(f"Error while setting device: {e}\nUsing cuda by default")
            self._device = torch.device("cuda")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def init(self, dataset: AbstractDataset):
        pass

    @abstractmethod
    def load(self, pcd_path: str, dataset: AbstractDataset):
        pass

    @abstractmethod
    def save(self, pcd_path: str):
        pass

    def iteration_start(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        pass

    def get_regularization_loss(self, camera: Camera) -> torch.Tensor:
        return torch.tensor(0.0, device=self._device)

    @abstractmethod
    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        # This function is used for rendering with optimization
        pass

    def render_forward_only(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        # This function is used for forward rendering, i.e. rendering without optimization
        return self.render(
            camera, GRsetting, GRzer
        )  # Default implementation. Can be overridden.

    def iteration_end(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

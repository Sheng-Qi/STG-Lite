from typing import Type
import torch
import logging
from abc import abstractmethod
from utils.graphics_utils import BasicPointCloud
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
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def init(self, pcd_data: BasicPointCloud):
        pass

    @abstractmethod
    def load(self, pcd_path: str):
        pass

    @abstractmethod
    def save(self, pcd_path: str):
        pass

    @abstractmethod
    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        # This function is used for rendering with optimization
        pass

    def render_forward(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        # This function is used for forward rendering, i.e. rendering without optimization
        return self.render(
            camera, GRsetting, GRzer
        )  # Default implementation. Should be overridden

    def iteration_start(self, iteration: int, camera: Camera):
        pass

    def iteration_end(self, iteration: int, camera: Camera):
        pass

    def get_regularization_loss(self, camera: Camera) -> torch.Tensor:
        return torch.tensor(0.0, device=self._device)

    @abstractmethod
    def __len__(self) -> int:
        pass

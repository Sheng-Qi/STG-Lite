from typing import Type
import torch
from abc import abstractmethod
from utils.graphics_utils import BasicPointCloud
from scene.cameras import Camera


class AbstractModel:
    @abstractmethod
    def __init__(self, model_params: dict, cameras_extent: float):
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def init_from_pcd(self, pcd_data: BasicPointCloud):
        pass

    @abstractmethod
    def load_from_pcd(self, pcd_path: str):
        pass

    @abstractmethod
    def save_pcd(self, pcd_path: str):
        pass

    @abstractmethod
    def iteration_start(self, iteration: int, camera: Camera):
        pass

    @abstractmethod
    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        pass

    @abstractmethod
    def render_forward(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        pass

    @abstractmethod
    def iteration_end(self, iteration: int, camera: Camera):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

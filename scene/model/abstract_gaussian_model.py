from typing import Type
import torch
from abc import abstractmethod
from scene.dataset.abstract_dataset import AbstractDataset
from scene.cameras import Camera


class AbstractModel:
    @abstractmethod
    def __init__(self, model_params: dict):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass
    
    @property
    @abstractmethod
    def sh_degree(self) -> int:
        pass

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

    @abstractmethod
    def iteration_start(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        pass

    @abstractmethod
    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        pass

    @abstractmethod
    def render_forward_only(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        pass

    @abstractmethod
    def get_regularization_loss(
        self, camera: Camera, dataset: AbstractDataset
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def one_up_sh_degree(self):
        pass

    @abstractmethod
    def iteration_end(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        pass

from abc import abstractmethod

from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud


class AbstractDataset:
    @abstractmethod
    def __init__(self, dataset_params: dict):
        pass

    @property
    @abstractmethod
    def train_cameras(self) -> list[Camera]:
        pass

    @property
    @abstractmethod
    def test_cameras(self) -> list[Camera]:
        pass

    @property
    @abstractmethod
    def ply_path(self) -> str:
        pass

    @property
    @abstractmethod
    def ply_data(self) -> BasicPointCloud:
        pass

    @property
    @abstractmethod
    def test_camera_index(self) -> int:
        pass

    @property
    @abstractmethod
    def nerf_norm(self) -> dict:
        pass

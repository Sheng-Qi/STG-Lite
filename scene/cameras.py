import numpy as np
import torch
import os
import logging
from dataclasses import dataclass
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getProjectionMatrixCV,
)


@dataclass
class CameraInfo:
    width: int
    height: int
    FovX: float
    FovY: float
    cxr: float
    cyr: float
    R: np.ndarray
    T: np.ndarray
    image_folder: str
    mask_folder: str
    image_name: str
    camera_id: int
    near: float
    far: float
    trans: np.ndarray
    scale: float
    timestamp: int
    timestamp_ratio: float


class Camera:

    def __init__(
        self,
        camera_info: CameraInfo,
        device: torch.device = torch.device("cuda"),
        data_device: str = "cpu",
        int8_mode: bool = True,
        resolution_scale: float = None,
        lazy_load: bool = True,
    ):
        self._camera_info = camera_info
        self._device = device
        try:
            self._data_device = torch.device(data_device)
        except Exception as e:
            logging.error(f"Error while setting device: {e}\nUsing CPU instead.")
            self._data_device = torch.device("cpu")
        self._int8_mode = int8_mode
        self._resolution_scale = resolution_scale
        self._lazy_load = lazy_load

        self._image = None
        self._image_mask = None
        self._world_view_transform = None
        self._projection_matrix = None
        self._full_proj_transform = None
        self._camera_center = None

        self._resized_resolution = None

        if not lazy_load:
            self._load_and_process_image()

    @property
    def resized_resolution(self) -> tuple[int, int]:
        if self._resized_resolution is None:
            if self._resolution_scale is None:
                if self._camera_info.width > 1600:
                    logging.warning(
                        "Encountered quite large input images, rescaling to 1.6K."
                    )
                    self._resolution_scale = self._camera_info.width / 1600
                else:
                    self._resolution_scale = 1.0

            self._resized_resolution = (
                round(self._camera_info.width / self._resolution_scale),
                round(self._camera_info.height / self._resolution_scale),
            )
        return self._resized_resolution

    @property
    def resized_width(self) -> int:
        return self.resized_resolution[0]

    @property
    def resized_height(self) -> int:
        return self.resized_resolution[1]

    @property
    def camera_info(self) -> CameraInfo:
        return self._camera_info

    @property
    def image(self) -> torch.Tensor:
        if self._image is None:
            self._load_and_process_image()
        if self._int8_mode:
            return self._image.to(self._device).float() / 255.0
        else:
            return self._image.to(self._device).float()

    @property
    def image_mask(self) -> torch.Tensor:
        if self._image is None:
            self._load_and_process_image()
        if self._image_mask is None:
            return torch.ones_like(self._image[0:1, ...], device=self._device)
        if self._int8_mode:
            return self._image_mask.to(self._device).float() / 255.0
        else:
            return self._image_mask.to(self._device).float()

    @property
    def world_view_transform(self) -> torch.Tensor:
        if self._world_view_transform is None:
            self._load_and_process_image()
        return self._world_view_transform

    @property
    def projection_matrix(self) -> torch.Tensor:
        if self._projection_matrix is None:
            self._load_and_process_image()
        return self._projection_matrix

    @property
    def full_proj_transform(self) -> torch.Tensor:
        if self._full_proj_transform is None:
            self._load_and_process_image()
        return self._full_proj_transform

    @property
    def camera_center(self) -> torch.Tensor:
        if self._camera_center is None:
            self._load_and_process_image()
        return self._camera_center

    def _load_and_process_image(self):
        image = Image.open(
            os.path.join(self._camera_info.image_folder, self._camera_info.image_name)
        )
        if (
            self._camera_info.width != image.width
            or self._camera_info.height != image.height
        ):
            raise ValueError("Image size does not match camera parameters")
        resize_image = PILtoTorch(image, resolution=self.resized_resolution)
        
        if self._camera_info.mask_folder is not None:
            image_mask = Image.open(
                os.path.join(
                    self._camera_info.mask_folder, self._camera_info.image_name
                )
            )
            if (
                self._camera_info.width != image_mask.width
                or self._camera_info.height != image_mask.height
            ):
                raise ValueError("Mask size does not match camera parameters")
            
            image_mask = image_mask.convert("L")
            resized_image_mask = PILtoTorch(image_mask, resolution=self.resized_resolution)
        elif resize_image.shape[0] == 4:
            resized_image_mask = resize_image[3:4, ...]
        else:
            # To save memory, we don't store the mask if it is not provided
            resized_image_mask = None

        resized_image_gt = resize_image[:3, ...]

        if self._int8_mode:
            if resized_image_mask is not None:
                self._image_mask = (resized_image_mask * 255).to(torch.uint8).to(self._data_device)
            else:
                self._image_mask = None
            self._image = (resized_image_gt * 255).to(torch.uint8).to(self._data_device)
        else:
            if resized_image_mask is not None:
                self._image_mask = resized_image_mask.clamp(0.0, 1.0).to(self._data_device)
            else:
                self._image_mask = None
            self._image = resized_image_gt.clamp(0.0, 1.0).to(self._data_device)

        self._world_view_transform = (
            torch.tensor(
                getWorld2View2(
                    self._camera_info.R,
                    self._camera_info.T,
                    self._camera_info.trans,
                    self._camera_info.scale,
                )
            )
            .transpose(0, 1)
            .to(self._device)
        )

        if self._camera_info.cyr != 0.0 and self._camera_info.cxr != 0.0:
            self._projection_matrix = (
                getProjectionMatrixCV(
                    znear=self._camera_info.near,
                    zfar=self._camera_info.far,
                    fovX=self._camera_info.FovX,
                    fovY=self._camera_info.FovY,
                    cx=self._camera_info.cxr,
                    cy=self._camera_info.cyr,
                )
                .clone()
                .detach()
                .transpose(0, 1)
                .to(self._device)
            )
        else:
            self._projection_matrix = (
                getProjectionMatrix(
                    znear=self._camera_info.near,
                    zfar=self._camera_info.far,
                    fovX=self._camera_info.FovX,
                    fovY=self._camera_info.FovY,
                )
                .clone()
                .detach()
                .transpose(0, 1)
                .to(self._device)
            )

        self._full_proj_transform = (
            self._world_view_transform.unsqueeze(0).bmm(
                self._projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self._camera_center = self._world_view_transform.inverse()[3, :3]

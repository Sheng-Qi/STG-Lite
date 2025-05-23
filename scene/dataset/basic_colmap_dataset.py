import os
import re
import numpy as np
import torch
from tqdm import tqdm
import logging
from typing import Optional
from pydantic import BaseModel, field_validator, Field

from scene.dataset.abstract_dataset import AbstractDataset
from scene.cameras import CameraInfo, Camera
from utils.colmap_utils import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
    Image as ColmapImage,
    Camera as ColmapCamera,
)
from utils.graphics_utils import focal2fov, getWorld2View2, BasicPointCloud
from utils.ply_utils import storePly3D, fetchPly3D


class BasicColmapDatasetParams(BaseModel):
    source_path: str
    relative_mask_path: Optional[str] = None
    resolution_scale: float = 2.0
    is_eval: bool = False
    train_test_split_seed: int = 0
    near: float = 0.1
    far: float = 100.0
    lazy_load: bool = True
    ply_path: Optional[str] = None
    data_device: str = "cpu"
    int8_mode: bool = True


class BasicColmapDatasetContext(BaseModel):
    device: str
    parallel_load: bool = True


class BasicColmapDataset(AbstractDataset):

    def __init__(self, dataset_params: dict, context: dict):
        self._dataset_params = BasicColmapDatasetParams.model_validate(dataset_params)
        self._context = BasicColmapDatasetContext(**context)
        self._train_cameras = None
        self._test_cameras = None
        self._camera_index_set = set()
        self._test_camera_index = None
        self._ply_path = self._dataset_params.ply_path
        self._ply_data = None

        if self._context.parallel_load and self._dataset_params.lazy_load:
            logging.warning("Parallel load is enabled but lazy load is disabled. Lazy load will not be working.")

    @property
    def near(self) -> float:
        return self._dataset_params.near

    @property
    def far(self) -> float:
        return self._dataset_params.far

    @property
    def train_cameras(self) -> list[Camera]:
        if self._train_cameras is None:
            self._load_cameras()
        return self._train_cameras

    @property
    def test_cameras(self) -> list[Camera]:
        if self._test_cameras is None:
            self._load_cameras()
        return self._test_cameras

    @property
    def nerf_norm(self) -> dict:
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = list()

        for cam in self.train_cameras:
            caminfo = cam.camera_info
            W2C = getWorld2View2(caminfo.R, caminfo.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return {"translate": translate, "radius": radius}

    @property
    def ply_path(self) -> str:
        if self.ply_data is None:
            self._load_ply()
        return self._ply_path

    @property
    def ply_data(self) -> BasicPointCloud:
        if self._ply_data is None:
            self._load_ply()
        return self._ply_data

    @property
    def test_camera_index(self) -> int:
        if self._test_camera_index is None:
            self._load_cameras()
        return self._test_camera_index

    def _load_cameras(self):
        cameras = self._read_colmap_cameras()
        if self._dataset_params.is_eval:
            if len(self._camera_index_set) == 1:
                logging.warning("Only one camera found. No train/test split will be performed.")
                self._train_cameras = cameras
                self._test_cameras = list[Camera]()
            else:
                self._train_cameras, self._test_cameras = self._train_test_split(cameras)
        else:
            self._train_cameras = cameras
            self._test_cameras = list[Camera]()

    def _load_ply(self):

        if self._ply_path is None:
            self._ply_path = os.path.join(
                self._dataset_params.source_path,
                "input_ply",
                f"points3D.ply",
            )
            logging.info("PLY path not provided. Setting to " + self._ply_path)
            if not os.path.exists(self._ply_path):
                logging.info(
                    "PLY file not found at "
                    + self._ply_path
                    + ". Creating from COLMAP files"
                )
                self._create_ply_from_colmap()
        elif not os.path.exists(self._ply_path):
            raise FileNotFoundError(f"PLY file not found at {self._ply_path}")
        else:
            logging.info("Reading PLY file from " + self._ply_path)
        self._ply_data = fetchPly3D(self._ply_path)

    def _read_colmap_cameras(self) -> list[Camera]:
        cameras = list[Camera]()
        if not os.path.exists(self._dataset_params.source_path):
            raise FileNotFoundError(
                f"Colmap folder not found at {self._dataset_params.source_path}"
            )

        cam_extrinsics, cam_intrinsics = self._find_and_read_colmap_files(
            self._dataset_params.source_path
        )

        for extrinsics in tqdm(
            cam_extrinsics.values(),
            desc="Reading cameras progress",
        ):
            intrinsics = cam_intrinsics[extrinsics.camera_id]

            width = intrinsics.width
            height = intrinsics.height
            if intrinsics.model == "PINHOLE":
                FovX = focal2fov(intrinsics.params[0], width)
                FovY = focal2fov(intrinsics.params[1], height)
                cxr = (intrinsics.params[2]) / width - 0.5
                cyr = (intrinsics.params[3]) / height - 0.5
            elif intrinsics.model == "SIMPLE_PINHOLE":
                FovX = focal2fov(intrinsics.params[0], width)
                FovY = focal2fov(intrinsics.params[0], height)
                cxr = (intrinsics.params[1]) / width - 0.5
                cyr = (intrinsics.params[1]) / height - 0.5
            else:
                raise NotImplementedError(
                    "Only PINHOLE and SIMPLE_PINHOLE models are supported"
                )

            R = np.transpose(qvec2rotmat(extrinsics.qvec))
            T = np.array(extrinsics.tvec)
            image_folder = os.path.join(self._dataset_params.source_path, "images")
            image_name = extrinsics.name

            if self._dataset_params.relative_mask_path is not None:
                mask_folder = os.path.join(
                    self._dataset_params.source_path,
                    self._dataset_params.relative_mask_path,
                )
            else:
                mask_folder = None

            camera_info = CameraInfo(
                width=width,
                height=height,
                FovX=FovX,
                FovY=FovY,
                cxr=cxr,
                cyr=cyr,
                R=R,
                T=T,
                image_folder=image_folder,
                mask_folder=mask_folder,
                image_name=image_name,
                camera_id=self._get_camera_index(image_name),
                near=self._dataset_params.near,
                far=self._dataset_params.far,
                trans=np.array([0, 0, 0]),
                scale=1.0,
                timestamp=None,
                timestamp_ratio=None,
            )
            should_load_later = self._dataset_params.lazy_load or self._context.parallel_load
            cameras.append(
                Camera(
                    camera_info,
                    device=self._context.device,
                    data_device=self._dataset_params.data_device,
                    int8_mode=self._dataset_params.int8_mode,
                    resolution_scale=self._dataset_params.resolution_scale,
                    should_load_later=should_load_later,
                )
            )

            camera_index = self._get_camera_index(image_name)
            self._camera_index_set.add(camera_index)
        return cameras

    def _train_test_split(
        self, cameras: list[Camera]
    ) -> tuple[list[Camera], list[Camera]]:
        camera_indices = list(self._camera_index_set)
        np.random.seed(self._dataset_params.train_test_split_seed)
        np.random.shuffle(camera_indices)
        self._test_camera_index = camera_indices[0]
        logging.info(f"Test camera index: {self._test_camera_index}")

        train_cameras = list[Camera]()
        test_cameras = list[Camera]()
        for camera in cameras:
            if (
                self._get_camera_index(camera.camera_info.image_name)
                == self._test_camera_index
            ):
                test_cameras.append(camera)
            else:
                train_cameras.append(camera)
        return train_cameras, test_cameras

    def _find_and_read_colmap_files(
        self, colmap_data_path: str
    ) -> tuple[dict[int, ColmapImage], dict[int, ColmapCamera]]:
        paths = [
            ("0/images.bin", "0/cameras.bin"),
            ("0/images.txt", "0/cameras.txt"),
            ("images.bin", "cameras.bin"),
            ("images.txt", "cameras.txt"),
        ]
        for images_file, cameras_file in paths:
            images_path = os.path.join(colmap_data_path, "sparse", images_file)
            cameras_path = os.path.join(colmap_data_path, "sparse", cameras_file)
            if os.path.exists(images_path) and os.path.exists(cameras_path):
                if images_file.endswith(".bin"):
                    cam_extrinsics = read_extrinsics_binary(images_path)
                    cam_intrinsics = read_intrinsics_binary(cameras_path)
                else:
                    cam_extrinsics = read_extrinsics_text(images_path)
                    cam_intrinsics = read_intrinsics_text(cameras_path)
                return cam_extrinsics, cam_intrinsics
        raise FileNotFoundError(f"Colmap files not found at {colmap_data_path}")

    def _get_camera_index(self, image_name: str) -> int:
        matches = [
            re.search(r"cam(\d+)[/_.]", image_name),
            re.search(r"(\d+)/(\d+)", image_name)
        ]
        for match in matches:
            if match:
                return int(match.group(1))
        logging.debug(f"Camera index not found in image name: {image_name}. Defaulting to 0. You might need to set density control period manually.")
        return 0

    def _create_ply_from_colmap(self):
        paths = ["0/points3D.bin", "0/points3D.txt", "points3D.bin", "points3D.txt"]

        for points3D_file in paths:
            points3D_path = os.path.join(
                self._dataset_params.source_path, "sparse", points3D_file
            )
            if os.path.exists(points3D_path):
                if points3D_file.endswith(".bin"):
                    xyz, rgb, _ = read_points3D_binary(points3D_path)
                else:
                    xyz, rgb, _ = read_points3D_text(points3D_path)
                storePly3D(self._ply_path, xyz, rgb)
                return
        raise FileNotFoundError(
            f"Colmap points3D files not found at {self._dataset_params.source_path}"
        )

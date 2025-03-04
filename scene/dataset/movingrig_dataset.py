import os
import re
import logging
import numpy as np
from pydantic import BaseModel, Field

from scene.dataset.basic_colmap_dataset import BasicColmapDataset
from scene.cameras import CameraInfo, Camera
from utils.colmap_utils import qvec2rotmat, read_points3D_binary, read_points3D_text
from utils.graphics_utils import focal2fov
from utils.ply_utils import storePly4D, fetchPly4D


class MovingRigDatasetParams(BaseModel):
    frame_rate: float = Field(..., gt=0)
    duration: int = Field(..., gt=0)


class MovingRigDataset(BasicColmapDataset):

    def __init__(self, dataset_params: dict, context: dict):
        time_params = dataset_params["time_params"]
        non_time_params = {
            k: v for k, v in dataset_params.items() if k != "time_params"
        }
        super().__init__(non_time_params, context)
        self._time_params = MovingRigDatasetParams(time_params)

    @property
    def frame_rate(self) -> float:
        return self._time_params.frame_rate

    @property
    def duration(self) -> int:
        return self._time_params.duration

    def _load_ply(self):
        if self._ply_path is None:
            self._ply_path = os.path.join(
                self._dataset_params.source_path,
                "input_ply",
                f"points3D.ply",
            )
            logging.info("PLY path not provided. Setting to " + self._ply_path)
            if not os.path.exists(self._ply_path):
                logging.warning(
                    "Creating PLY file from colmap points3D.txt is not currently well supported in MovingRigDataset."
                )
                logging.warning("Please create the PLY file manually.")
                self._create_ply_from_colmap()
        elif not os.path.exists(self._ply_path):
            raise FileNotFoundError(f"PLY file not found at {self._ply_path}")
        else:
            logging.info("Reading PLY file from " + self._ply_path)
        self._ply_data = fetchPly4D(self._ply_path)

    def _read_colmap_cameras(self) -> list[Camera]:
        cameras = list[Camera]()
        cam_extrinsics, cam_intrinsics = self._find_and_read_colmap_files(
            self._dataset_params.source_path,
        )
        for extrinsics in cam_extrinsics.values():
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
                mask_folder = os.path.join(self._dataset_params.source_path, self._dataset_params.relative_mask_path)
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
                timestamp=self._parse_time(image_name),
                timestamp_ratio=self._parse_time(image_name)
                / self._time_params.duration,
            )
            cameras.append(
                Camera(
                    camera_info,
                    device=self._context.device,
                    data_device=self._dataset_params.data_device,
                    int8_mode=self._dataset_params.int8_mode,
                    resolution_scale=self._dataset_params.resolution_scale,
                    lazy_load=self._dataset_params.lazy_load,
                )
            )

            camera_index = self._get_camera_index(image_name)
            self._camera_index_set.add(camera_index)
        return cameras

    def _create_ply_from_colmap(self):
        paths = [
            os.path.join(self._dataset_params.source_path, "sparse", "0", "points3D.bin"),
            os.path.join(self._dataset_params.source_path, "sparse", "0", "points3D.txt"),
            os.path.join(self._dataset_params.source_path, "sparse", "points3D.bin"),
            os.path.join(self._dataset_params.source_path, "sparse", "points3D.txt"),
        ]
        for path in paths:
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(
                "Could not find points3D.bin or points3D.txt in the sparse folder"
            )
        if path.endswith(".bin"):
            xyz, rgb, _ = read_points3D_binary(path)
        else:
            xyz, rgb, _ = read_points3D_text(path)
        xyzt = np.concatenate((xyz, np.full_like(xyz[:, :1], 0.5)), axis=1)
        storePly4D(self._ply_path, xyzt, rgb)

    def _parse_time(self, image_name):
        match = re.match(r"cam(\d+)/cam(\d+)_(\d+).png", image_name)
        if match is None:
            raise ValueError("Invalid image name")
        frame_index = int(match.group(3))
        return frame_index

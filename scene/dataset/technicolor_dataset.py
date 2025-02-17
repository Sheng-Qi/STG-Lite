import os
import re
import numpy as np
from tqdm import tqdm
import logging

from scene.dataset.basic_colmap_dataset import BasicColmapDataset
from scene.cameras import CameraInfo, Camera
from utils.colmap_utils import (
    qvec2rotmat,
    read_points3D_binary,
    read_points3D_text
)
from utils.graphics_utils import focal2fov
from utils.ply_utils import storePly4D, fetchPly4D


class TechnicolorDataset(BasicColmapDataset):

    def __init__(self, dataset_params: dict):
        super().__init__(dataset_params)
        self._frame_rate: float = dataset_params["frame_rate"]
        self._start_frame: int = dataset_params["start_frame"]
        self._duration: int = dataset_params["duration"]

    def _load_ply(self):
        if self._ply_path is None:
            self._ply_path = os.path.join(
                self._source_path,
                "input_ply",
                f"points3D_{self._start_frame}_{self._start_frame + self._duration}.ply",
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
        self._ply_data = fetchPly4D(self._ply_path)

    def _read_colmap_cameras(self) -> list[Camera]:
        cameras = list[Camera]()
        for time in tqdm(
            range(self._start_frame, self._start_frame + self._duration),
            desc="Reading cameras progress",
        ):
            subfolder_path = os.path.join(self._source_path, f"colmap_{time}")
            if not os.path.exists(subfolder_path):
                raise FileNotFoundError(f"Colmap folder not found at {subfolder_path}")

            cam_extrinsics, cam_intrinsics = self._find_and_read_colmap_files(
                subfolder_path
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
                image_folder = os.path.join(subfolder_path, "images")
                image_name = extrinsics.name

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
                    mask_folder=None,
                    image_name=image_name,
                    camera_id=self._get_camera_index(image_name),
                    near=self._near,
                    far=self._far,
                    trans=np.array([0, 0, 0]),
                    scale=1.0,
                    timestamp=time,
                    timestamp_ratio=(time - self._start_frame) / self._duration,
                )
                cameras.append(
                    Camera(
                        camera_info,
                        device=self._device,
                        data_device=self._data_device,
                        int8_mode=self._int8_mode,
                        resolution_scale=self._resolution_scale,
                        lazy_load=self._lazy_load,
                    )
                )

                camera_index = self._get_camera_index(image_name)
                self._camera_index_set.add(camera_index)

        return cameras

    def _create_ply_from_colmap(self):
        totalxyz = list()
        totalrgb = list()
        totaltime = list()

        for time in range(self._start_frame, self._start_frame + self._duration):
            current_time_bin_path = os.path.join(
                self._source_path, f"colmap_{time}", "sparse", "0", "points3D.bin"
            )
            current_time_txt_path = os.path.join(
                self._source_path, f"colmap_{time}", "sparse", "0", "points3D.txt"
            )
            if os.path.exists(current_time_bin_path):
                xyz, rgb, _ = read_points3D_binary(current_time_bin_path)
            else:
                xyz, rgb, _ = read_points3D_text(current_time_txt_path)

            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(
                np.ones((xyz.shape[0], 1)) * (time - self._start_frame) / self._duration
            )
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly4D(self._ply_path, xyzt, rgb)

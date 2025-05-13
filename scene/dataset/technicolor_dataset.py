import os
import re
import numpy as np
from tqdm import tqdm
import logging
from pydantic import BaseModel, Field

from scene.dataset.basic_colmap_dataset import BasicColmapDataset
from scene.cameras import CameraInfo, Camera
from utils.colmap_utils import qvec2rotmat, read_points3D_binary, read_points3D_text
from utils.graphics_utils import focal2fov
from utils.ply_utils import storePly4D, fetchPly4D


class TechinicolorDatasetParams(BaseModel):
    frame_rate: float = Field(..., gt=0)
    start_frame: int = Field(..., ge=0)
    frame_count: int = Field(..., gt=0)


class TechnicolorDataset(BasicColmapDataset):

    def __init__(self, dataset_params: dict, context: dict):
        time_params = dataset_params["time_params"]
        non_time_params = {
            k: v for k, v in dataset_params.items() if k != "time_params"
        }
        super().__init__(non_time_params, context)
        self._time_params = TechinicolorDatasetParams.model_validate(time_params)

    @property
    def frame_rate(self) -> float:
        return self._time_params.frame_rate

    @property
    def start_frame(self) -> int:
        return self._time_params.start_frame

    @property
    def frame_count(self) -> int:
        return self._time_params.frame_count

    def _load_ply(self):
        if self._ply_path is None:
            self._ply_path = os.path.join(
                self._dataset_params.source_path,
                "input_ply",
                f"points3D_{self._time_params.start_frame}_{self._time_params.start_frame + self._time_params.frame_count}.ply",
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
            range(
                self._time_params.start_frame,
                self._time_params.start_frame + self._time_params.frame_count,
            ),
            desc="Reading cameras progress",
        ):
            all_subfolders = os.listdir(self._dataset_params.source_path)
            subfolder_path = None
            for subfolder in all_subfolders:
                if re.match(r"colmap_0*{}$".format(time), subfolder):
                    subfolder_path = os.path.join(
                        self._dataset_params.source_path, subfolder
                    )
                    break
            if subfolder_path is None:
                raise FileNotFoundError(
                    f"Colmap folder not found at time {time}"
                )

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

                if self._dataset_params.relative_mask_path is not None:
                    mask_folder = os.path.join(
                        subfolder_path,
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
                    timestamp=time,
                    timestamp_ratio=(time - self._time_params.start_frame)
                    / self._time_params.frame_count,
                )
                cameras.append(
                    Camera(
                        camera_info,
                        device=self._context.device,
                        data_device=self._dataset_params.data_device,
                        int8_mode=self._dataset_params.int8_mode,
                        resolution_scale=self._dataset_params.resolution_scale,
                        should_load_later=self._dataset_params.lazy_load,
                    )
                )

                camera_index = self._get_camera_index(image_name)
                self._camera_index_set.add(camera_index)

        return cameras

    def _create_ply_from_colmap(self):
        totalxyz = list()
        totalrgb = list()
        totaltime = list()

        for time in range(
            self._time_params.start_frame,
            self._time_params.start_frame + self._time_params.frame_count,
        ):
            all_subfolders = os.listdir(self._dataset_params.source_path)
            subfolder_path = None
            for subfolder in all_subfolders:
                if re.match(r"colmap_0*{}$".format(time), subfolder):
                    subfolder_path = os.path.join(
                        self._dataset_params.source_path, subfolder
                    )
                    break
            if subfolder_path is None:
                raise FileNotFoundError(
                    f"Colmap folder not found at time {time}"
                )
            paths = [
                os.path.join(
                    self._dataset_params.source_path,
                    subfolder_path,
                    "sparse",
                    "0",
                    "points3D.bin",
                ),
                os.path.join(
                    self._dataset_params.source_path,
                    subfolder_path,
                    "sparse",
                    "0",
                    "points3D.txt",
                ),
                os.path.join(
                    self._dataset_params.source_path,
                    subfolder_path,
                    "sparse",
                    "points3D.bin",
                ),
                os.path.join(
                    self._dataset_params.source_path,
                    subfolder_path,
                    "sparse",
                    "points3D.txt",
                ),
            ]
            for path in paths:
                if os.path.exists(path):
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find points3D file for time {time} at any of the following paths: {paths}"
                )
            if path.endswith(".bin"):
                xyz, rgb, _ = read_points3D_binary(path)
            else:
                xyz, rgb, _ = read_points3D_text(path)

            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(
                np.ones((xyz.shape[0], 1))
                * (time - self._time_params.start_frame)
                / self._time_params.frame_count
            )
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        xyzt = np.concatenate((xyz, totaltime), axis=1)
        storePly4D(self._ply_path, xyzt, rgb)

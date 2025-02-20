import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData

from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid


class Spacetime360Model(SpacetimeGaussianModel):
    def __init__(self, config):
        super().__init__(config)

        self._static_ply_path: str = config["static_ply_path"]
        self._border_image_threshold: int = config["border_image_threshold"]
        self._enable_simplified_border_method: bool = config[
            "enable_simplified_border_method"
        ]

        self._static_xyz: torch.Tensor = None
        self._static_xyz_scales: torch.Tensor = None
        self._static_rotation: torch.Tensor = None
        self._static_opacity: torch.Tensor = None
        self._static_features_dc: torch.Tensor = None
        self._static_t: torch.Tensor = None
        self._STATIC_T_SCALE = 1e5
        self._static_motion: torch.Tensor = None
        self._static_omega: torch.Tensor = None

        self._static_radii: torch.Tensor = None
        self._static_screenspace_points: torch.Tensor = None

        self._train_cameras = None
        self._camset = None
        self._cached_count_in_cameras = None

    @property
    def xyz(self) -> torch.Tensor:
        return torch.cat(
            [
                super().xyz,
                torch.where(
                    self._center_image_mask, self._static_xyz, self._static_xyz.detach()
                ),
            ],
            dim=0,
        )

    @property
    def xyz_scales(self) -> torch.Tensor:
        return torch.cat(
            [
                super().xyz_scales,
                self._static_xyz_scales.detach(),
            ],
            dim=0,
        )

    @property
    def xyz_scales_active(self) -> torch.Tensor:
        return torch.cat(
            [
                super().xyz_scales_active,
                torch.where(
                    self._center_image_mask,
                    torch.exp(self._static_xyz_scales),
                    torch.exp(self._static_xyz_scales).detach(),
                ),
            ],
            dim=0,
        )

    @property
    def rotation(self) -> torch.Tensor:
        return torch.cat(
            [
                super().rotation,
                torch.where(
                    self._center_image_mask,
                    self._static_rotation,
                    self._static_rotation.detach(),
                ),
            ],
            dim=0,
        )

    @property
    def rotation_active(self) -> torch.Tensor:
        return torch.cat(
            [
                super().rotation_active,
                torch.where(
                    self._center_image_mask,
                    torch.nn.functional.normalize(self._static_rotation),
                    torch.nn.functional.normalize(self._static_rotation).detach(),
                ),
            ],
            dim=0,
        )

    @property
    def opacity(self) -> torch.Tensor:
        return torch.cat(
            [
                super().opacity,
                self._static_opacity,
            ],
            dim=0,
        )

    @property
    def opacity_active(self) -> torch.Tensor:
        return torch.cat(
            [
                super().opacity_active,
                torch.sigmoid(self._static_opacity),
            ],
            dim=0,
        )

    @property
    def features_dc(self) -> torch.Tensor:
        return torch.cat(
            [super().features_dc, self._static_features_dc.detach()],
            dim=0,
        )

    @property
    def t(self) -> torch.Tensor:
        return torch.cat(
            [
                super().t,
                torch.where(
                    self._center_image_mask,
                    self._static_t,
                    self._static_t.detach(),
                ),
            ],
            dim=0,
        )

    @property
    def t_scale(self) -> torch.Tensor:
        return torch.cat(
            [
                super().t_scale,
                torch.full_like(
                    self._static_t, torch.exp(torch.tensor(self._STATIC_T_SCALE))
                ),
            ],
            dim=0,
        )

    @property
    def t_scale_active(self) -> torch.Tensor:
        return torch.cat(
            [
                super().t_scale_active,
                torch.full_like(
                    self._static_t, torch.exp(torch.tensor(self._STATIC_T_SCALE))
                ),
            ],
            dim=0,
        )

    @property
    def motion(self) -> torch.Tensor:
        return torch.cat(
            [
                super().motion,
                torch.where(
                    self._center_image_mask,
                    self._static_motion,
                    self._static_motion.detach(),
                ),
            ],
            dim=0,
        )

    @property
    def omega(self) -> torch.Tensor:
        return torch.cat(
            [
                super().omega,
                torch.where(
                    self._center_image_mask,
                    self._static_omega,
                    self._static_omega.detach(),
                ),
            ],
            dim=0,
        )

    @property
    def _count_in_cameras(self) -> torch.Tensor:
        if self._cached_count_in_cameras is None:
            self._update_count_in_cameras()
        return self._cached_count_in_cameras

    @property
    def _center_image_mask(self) -> torch.Tensor:
        return (self._count_in_cameras > self._border_image_threshold).unsqueeze(1)

    def init(self, dataset):
        super().init(dataset)
        self._train_cameras = dataset.train_cameras
        self._camset = {
            camera.camera_info.camera_id: camera for camera in self._train_cameras
        }

    def load(self, pcd_path, dataset):
        super().load(pcd_path, dataset)
        self._train_cameras = dataset.train_cameras
        self._camset = {
            camera.camera_info.camera_id: camera for camera in self._train_cameras
        }

    def iteration_start(self, iteration, camera, dataset):
        super().iteration_start(iteration, camera, dataset)
        self._update_count_in_cameras()

    def _update_count_in_cameras(self):
        self._cached_count_in_cameras = torch.zeros_like(
            self._static_xyz[:, 0], device=self._device
        )
        if self._enable_simplified_border_method:
            for camera in self._camset.values():
                self._cached_count_in_cameras += camera.check_in_image(self._static_xyz)
            self._cached_count_in_cameras = (
                self._cached_count_in_cameras
                / len(self._camset)
                * len(self._train_cameras)
            )
        else:
            for camera in self._train_cameras:
                self._cached_count_in_cameras += camera.check_in_image(self._static_xyz)

    def _init_point_cloud_parameters(self, pcd_data: BasicPointCloud):
        super()._init_point_cloud_parameters(pcd_data)
        self._init_static_parameters(pcd_data)

    def _fetch_point_cloud_parameters(self, ply_data, init_pcd_data):
        super()._fetch_point_cloud_parameters(ply_data, init_pcd_data)
        self._init_static_parameters(init_pcd_data)

    def _init_static_parameters(self, dynamic_pcd_data: BasicPointCloud):
        def SH2RGB(sh):
            C0 = 0.28209479177387814
            return sh * C0 + 0.5

        static_ply_data = PlyData.read(self._static_ply_path)

        xyz_names = ["x", "y", "z"]
        xyz_scales_names = ["scale_" + str(i) for i in range(3)]
        rotation_names = ["rot_" + str(i) for i in range(4)]
        opacity_names = ["opacity"]
        features_dc_names = ["f_dc_" + str(i) for i in range(3)]

        xyz = np.stack(
            [np.asarray(static_ply_data.elements[0][name]) for name in xyz_names],
            axis=1,
        )
        xyz_scales = np.stack(
            [
                np.asarray(static_ply_data.elements[0][name])
                for name in xyz_scales_names
            ],
            axis=1,
        )
        rotation = np.stack(
            [np.asarray(static_ply_data.elements[0][name]) for name in rotation_names],
            axis=1,
        )
        opacity = np.stack(
            [np.asarray(static_ply_data.elements[0][name]) for name in opacity_names],
            axis=1,
        )
        features_dc = np.stack(
            [
                np.asarray(static_ply_data.elements[0][name])
                for name in features_dc_names
            ],
            axis=1,
        )

        self._static_xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self._device).requires_grad_(
                True
            )
        )
        self._static_xyz_scales = nn.Parameter(
            torch.tensor(
                xyz_scales, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._static_rotation = nn.Parameter(
            torch.tensor(
                rotation, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._static_opacity = nn.Parameter(
            torch.tensor(
                opacity, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._static_features_dc = nn.Parameter(
            torch.tensor(
                SH2RGB(features_dc), dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._static_t = nn.Parameter(
            torch.zeros(
                self._static_xyz.shape[0], self._t.shape[1], device=self._device
            ).requires_grad_(True)
        )
        self._static_motion = nn.Parameter(
            torch.zeros(
                self._static_xyz.shape[0], self._motion.shape[1], device=self._device
            ).requires_grad_(True)
        )
        self._static_omega = nn.Parameter(
            torch.zeros(
                self._static_xyz.shape[0], self._omega.shape[1], device=self._device
            ).requires_grad_(True)
        )

        point_cloud_dynamic = (
            torch.tensor(np.asarray(dynamic_pcd_data.points)).float().to(self._device)
        )
        self._init_outline_bounds(
            torch.cat([point_cloud_dynamic, self._static_xyz], dim=0)
        )  # reinit outline bounds

    @property
    def _initial_param_groups(self) -> list:
        static_param_groups = [
            {
                "params": [self._static_xyz],
                "lr": self._learning_rate["xyz_init"] * self._cameras_extent,
                "name": "static_xyz",
            },
            {
                "params": [self._static_xyz_scales],
                "lr": self._learning_rate["xyz_scales"],
                "name": "static_xyz_scales",
            },
            {
                "params": [self._static_rotation],
                "lr": self._learning_rate["rotation"],
                "name": "static_rotation",
            },
            {
                "params": [self._static_opacity],
                "lr": self._learning_rate["opacity"],
                "name": "static_opacity",
            },
            {
                "params": [self._static_t],
                "lr": self._learning_rate["t"],
                "name": "static_t",
            },
            {
                "params": [self._static_motion],
                "lr": self._learning_rate["xyz_init"]
                * self._cameras_extent
                * 0.5
                * self._learning_rate["motion"],
                "name": "static_motion",
            },
            {
                "params": [self._static_omega],
                "lr": self._learning_rate["omega"],
                "name": "static_omega",
            },
            {
                "params": [self._static_features_dc],
                "lr": self._learning_rate["features_dc"],
                "name": "static_features_dc",
            }
        ]
        return super()._initial_param_groups + static_param_groups

    def _cache_density_control(self):
        if self._radii is None:
            raise ValueError(
                "Radii not computed, please set densification_start to a positive value"
            )
        radii_dynamic = self._radii[: self._xyz.shape[0]]
        self._max_radii2D[radii_dynamic > 0] = torch.max(
            self._max_radii2D[radii_dynamic > 0],
            radii_dynamic[radii_dynamic > 0],
        )
        self._max_vspace_gradient[radii_dynamic > 0] = torch.max(
            self._max_vspace_gradient[radii_dynamic > 0],
            torch.norm(
                self._screenspace_points.grad[: self._xyz.shape[0]][
                    radii_dynamic > 0, :2
                ],
                dim=-1,
                keepdim=True,
            ),
        )
        self._density_control_denom[radii_dynamic > 0] += 1
        if self._enable_time_density:
            self._max_vtime_gradient[radii_dynamic > 0] = torch.max(
                self._max_vtime_gradient[radii_dynamic > 0],
                self._t.grad[radii_dynamic > 0],
            )
            self._max_t_scale_active[radii_dynamic > 0] = torch.max(
                self._max_t_scale_active[radii_dynamic > 0],
                torch.exp(self._t_scale[radii_dynamic > 0]),
            )

    def _update_learning_rate(self, iteration: int):
        super()._update_learning_rate(iteration)
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "static_xyz":
                param_group["lr"] = self._xyz_scheduler_args(iteration)

    def _reset_opacity(self):
        super()._reset_opacity()
        new_opacity = inverse_sigmoid(
            0.1
            * torch.ones(
                (self._static_xyz.shape[0], 1), dtype=torch.float, device=self._device
            )
        )
        masked_opacity = torch.where(
            (self._count_in_cameras > 0).unsqueeze(1), new_opacity, self._static_opacity
        )
        self._static_opacity = self._reset_param_group(
            {"static_opacity": masked_opacity}
        )["static_opacity"]

    def _remove_outlier(self):
        super()._remove_outlier()
        max_bounds = torch.tensor(self._max_bounds_init, device=self._device)
        min_bounds = torch.tensor(self._min_bounds_init, device=self._device)
        mask_max = torch.any(self._static_xyz > max_bounds, dim=1)
        mask_min = torch.any(self._static_xyz < min_bounds, dim=1)
        mask = torch.logical_or(mask_max, mask_min)
        self._prune_points_static(mask)

    def _filter_invisible(self):
        super()._filter_invisible()
        self._prune_points_static(
            torch.sigmoid(self._static_opacity) < self._prune_threshold
        )

    def _prune_points_static(self, mask: torch.Tensor):
        mask = mask.squeeze()
        valid_point_mask = torch.logical_not(mask)
        param_list = [
            "static_xyz",
            "static_t",
            "static_xyz_scales",
            "static_rotation",
            "static_motion",
            "static_omega",
            "static_opacity",
            "static_features_dc",
        ]
        param_dict = {param: valid_point_mask for param in param_list}
        gaussian_optimizable_tensors = self._prune_param_group(param_dict)
        self._static_xyz = gaussian_optimizable_tensors["static_xyz"]
        self._static_t = gaussian_optimizable_tensors["static_t"]
        self._static_xyz_scales = gaussian_optimizable_tensors["static_xyz_scales"]
        self._static_rotation = gaussian_optimizable_tensors["static_rotation"]
        self._static_motion = gaussian_optimizable_tensors["static_motion"]
        self._static_omega = gaussian_optimizable_tensors["static_omega"]
        self._static_opacity = gaussian_optimizable_tensors["static_opacity"]
        self._static_features_dc = gaussian_optimizable_tensors["static_features_dc"]

        torch.cuda.empty_cache()

    def __len__(self) -> int:
        return super().__len__() + self._static_xyz.shape[0]

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Type
import math
from plyfile import PlyData, PlyElement

from scene.model.basic_gaussian_model import BasicGaussianModel
from scene.cameras import Camera
from utils.math_utils import trbfunction
from utils.general_utils import get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud


class SpacetimeGaussianModel(BasicGaussianModel):

    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self._t_scale_init: float = model_params["t_scale_init"]

        self._t = None
        self._t_scale = None
        self._motion = None
        self._omega = None

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @property
    def t_scale_active(self) -> torch.Tensor:
        return torch.exp(self._t_scale)

    @property
    def motion(self) -> torch.Tensor:
        return self._motion

    @property
    def omega(self) -> torch.Tensor:
        return self._omega

    def xyz_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return (
            self._xyz
            + self._motion[:, :3] * delta_t
            + self._motion[:, 3:6] * delta_t**2
            + self._motion[:, 6:] * delta_t**3
        )

    def rotation_active_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation + delta_t * self._omega)

    def opacity_active_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return self.opacity_active * trbfunction(delta_t / self.t_scale_active)

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        self._screenspace_points = torch.zeros_like(
            self._xyz, dtype=self._xyz.dtype, requires_grad=True, device=self._device
        )
        self._screenspace_points.retain_grad()
        raster_settings = GRsetting(
            image_height=int(camera.resized_height),
            image_width=int(camera.resized_width),
            tanfovx=math.tan(camera.camera_info.FovX * 0.5),
            tanfovy=math.tan(camera.camera_info.FovY * 0.5),
            bg=self._background_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)

        delta_t = (
            torch.full(
                (self._xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=self._xyz.dtype,
                requires_grad=False,
                device=self._device,
            )
            - self._t
        )

        self._rendered_image, self._radii, self._rendered_depth = rasterizer(
            means3D=self.xyz_projected(delta_t),
            means2D=self._screenspace_points,
            shs=None,
            colors_precomp=self._features_dc,
            opacities=self.opacity_active_projected(delta_t),
            scales=torch.exp(self._xyz_scales),
            rotations=self.rotation_active_projected(delta_t),
            cov3D_precomp=None,
        )

        if self._enable_color_transform:
            self._rendered_image = self._apply_color_transformation(
                camera, self._rendered_image
            )

        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def render_forward_only(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        self._screenspace_points = torch.zeros_like(
            self._xyz, dtype=self._xyz.dtype, requires_grad=True, device=self._device
        )
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        raster_settings = GRsetting(
            image_height=int(camera.resized_height),
            image_width=int(camera.resized_width),
            tanfovx=math.tan(camera.camera_info.FovX * 0.5),
            tanfovy=math.tan(camera.camera_info.FovY * 0.5),
            bg=self._background_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)
        delta_t = (
            torch.full(
                (self._xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=self._xyz.dtype,
                requires_grad=False,
                device=self._device,
            )
            - self._t
        )
        rendered_image, radii = rasterizer(
            timestamp=camera.camera_info.timestamp_ratio,
            trbfcenter=self._t,
            trbfscale=torch.exp(self._t_scale),
            motion=self._motion,
            means3D=self._xyz,
            means2D=self._screenspace_points,
            shs=None,
            colors_precomp=self._features_dc,
            opacities=torch.sigmoid(self._opacity),
            scales=torch.exp(self._xyz_scales),
            rotations=self._get_rotation_projected(delta_t),
            cov3D_precomp=None,
        )

        if self._enable_color_transform:
            rendered_image = self._apply_color_transformation_forward_only(
                camera, rendered_image, radii
            )

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        return {
            "rendered_image": rendered_image,
            "radii": radii,
            "duration": duration,
        }

    def _init_point_cloud_parameters(self, pcd_data: BasicPointCloud):
        super()._init_point_cloud_parameters(pcd_data)
        times = torch.tensor(np.asarray(pcd_data.times)).float().to(self._device)

        self._t = nn.Parameter(times.contiguous().requires_grad_(True))
        self._t_scale = nn.Parameter(
            torch.full(
                (self._t.shape[0], 1),
                self._t_scale_init,
                device=self._device,
                requires_grad=True,
            )
        )
        self._motion = nn.Parameter(
            torch.zeros((self._t.shape[0], 9), device=self._device).requires_grad_(True)
        )
        self._omega = nn.Parameter(
            torch.zeros((self._t.shape[0], 4), device=self._device).requires_grad_(True)
        )

    def _fetch_point_cloud_parameters(
        self, ply_data: PlyData, init_pcd_data: BasicPointCloud
    ):
        super()._fetch_point_cloud_parameters(ply_data, init_pcd_data)
        t_names = ["trbf_center"]
        t_scale_names = ["trbf_scale"]
        motion_names = ["motion_" + str(i) for i in range(9)]
        omega_names = ["omega_" + str(i) for i in range(4)]

        t = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in t_names], axis=1
        )
        t_scale = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in t_scale_names], axis=1
        )
        motion = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in motion_names], axis=1
        )
        omega = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in omega_names], axis=1
        )

        self._t = nn.Parameter(
            torch.tensor(t, dtype=torch.float, device=self._device).requires_grad_(True)
        )
        self._t_scale = nn.Parameter(
            torch.tensor(
                t_scale, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._motion = nn.Parameter(
            torch.tensor(motion, dtype=torch.float, device=self._device).requires_grad_(
                True
            )
        )
        self._omega = nn.Parameter(
            torch.tensor(omega, dtype=torch.float, device=self._device).requires_grad_(
                True
            )
        )

    def _save_point_cloud_parameters(self, pcd_path):
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        trbf_center = self._t.detach().cpu().numpy()
        trbf_scale = self._t_scale.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        motion = self._motion.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._xyz_scales.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()

        list_of_attributes = []
        list_of_attributes.extend(["x", "y", "z"])
        list_of_attributes.extend(["trbf_center"])
        list_of_attributes.extend(["trbf_scale"])
        list_of_attributes.extend(["nx", "ny", "nz"])
        list_of_attributes.extend(["motion_" + str(i) for i in range(9)])
        list_of_attributes.extend(["f_dc_" + str(i) for i in range(3)])
        list_of_attributes.extend(["opacity"])
        list_of_attributes.extend(["scale_" + str(i) for i in range(3)])
        list_of_attributes.extend(["rot_" + str(i) for i in range(4)])
        list_of_attributes.extend(["omega_" + str(i) for i in range(4)])

        dtype_full = [(attribute, "f4") for attribute in list_of_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                trbf_center,
                trbf_scale,
                normals,
                motion,
                f_dc,
                opacity,
                scale,
                rotation,
                omega,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        pcd_vertex_element = PlyElement.describe(elements, "vertex")
        PlyData([pcd_vertex_element]).write(pcd_path)

    def _initialize_learning_rate(self):
        l = [
            {
                "params": [self._xyz],
                "lr": self._learning_rate["xyz_init"] * self._cameras_extent,
                "name": "xyz",
            },
            {
                "params": [self._t],
                "lr": self._learning_rate["t"],
                "name": "t",
            },
            {
                "params": [self._xyz_scales],
                "lr": self._learning_rate["xyz_scales"],
                "name": "xyz_scales",
            },
            {
                "params": [self._t_scale],
                "lr": self._learning_rate["t_scale"],
                "name": "t_scale",
            },
            {
                "params": [self._rotation],
                "lr": self._learning_rate["rotation"],
                "name": "rotation",
            },
            {
                "params": [self._motion],
                "lr": self._learning_rate["xyz_init"]
                * self._cameras_extent
                * 0.5
                * self._learning_rate["motion"],
                "name": "motion",
            },
            {
                "params": [self._omega],
                "lr": self._learning_rate["omega"],
                "name": "omega",
            },
            {
                "params": [self._opacity],
                "lr": self._learning_rate["opacity"],
                "name": "opacity",
            },
            {
                "params": [self._features_dc],
                "lr": self._learning_rate["features_dc"],
                "name": "features_dc",
            },
        ]
        self._optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self._xyz_scheduler_args = get_expon_lr_func(
            lr_init=self._learning_rate["xyz_init"] * self._cameras_extent,
            lr_final=self._learning_rate["xyz_final"] * self._cameras_extent,
            lr_delay_steps=self._learning_rate["xyz_delay_steps"],
            lr_delay_mult=self._learning_rate["xyz_delay_mult"],
            max_steps=self._learning_rate["xyz_max_steps"],
        )
        self._color_transformation_scheduler_args = get_expon_lr_func(
            lr_init=self._learning_rate["color_transformation_init"],
            lr_final=self._learning_rate["color_transformation_final"],
            lr_delay_steps=self._learning_rate["color_transformation_delay_steps"],
            lr_delay_mult=self._learning_rate["color_transformation_delay_mult"],
            max_steps=self._learning_rate["color_transformation_max_steps"],
        )

    def _density_control(self, iteration: int):
        if iteration in range(self._densification_start, self._densification_end):
            self._cache_density_control()
            if (iteration + 1) in range(
                self._densification_start,
                self._densification_end,
                self._densification_step,
            ):
                space_mask = (
                    self._max_vspace_gradient.squeeze()
                    * torch.pow(
                        self._density_control_denom.squeeze()
                        / self._densification_step,
                        2,
                    )
                    * self._max_radii2D.squeeze()
                    * torch.sqrt(torch.sigmoid(self._opacity).squeeze())
                    > self._grad_threshold_xyz
                )
                space_mask = torch.logical_and(
                    space_mask, torch.sigmoid(self._opacity).squeeze() > 0.15
                )

                space_split_mask = torch.logical_and(
                    space_mask,
                    torch.max(torch.exp(self._xyz_scales), dim=1).values
                    > self._percent_dense_xyz * self._cameras_extent,
                )
                space_split_mask_under_selection = torch.logical_and(
                    space_split_mask, space_mask
                )[space_mask].repeat(self._split_num)

                new_xyz = self._xyz[space_mask].repeat(self._split_num, 1)
                new_t = self._t[space_mask].repeat(self._split_num, 1)
                new_xyz_scales = self._xyz_scales[space_mask].repeat(self._split_num, 1)
                new_t_scale = self._t_scale[space_mask].repeat(self._split_num, 1)
                new_rotation = self._rotation[space_mask].repeat(self._split_num, 1)
                new_motion = self._motion[space_mask].repeat(self._split_num, 1)
                new_omega = self._omega[space_mask].repeat(self._split_num, 1)
                new_opacity = self._opacity[space_mask].repeat(self._split_num, 1)
                new_features_dc = self._features_dc[space_mask].repeat(
                    self._split_num, 1
                )

                stds = torch.exp(self._xyz_scales[space_split_mask]).repeat(
                    self._split_num, 1
                )
                means = torch.zeros((stds.size(0), 3), device=self._device)
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[space_split_mask]).repeat(
                    self._split_num, 1, 1
                )
                new_xyz_random_offset = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(
                    -1
                )
                new_xyz[space_split_mask_under_selection] += new_xyz_random_offset

                new_xyz_scales[space_split_mask_under_selection] = torch.log(
                    torch.exp(new_xyz_scales[space_split_mask_under_selection])
                    * self._split_ratio
                )

                param_dict = {
                    "xyz": new_xyz,
                    "t": new_t,
                    "xyz_scales": new_xyz_scales,
                    "t_scale": new_t_scale,
                    "rotation": new_rotation,
                    "motion": new_motion,
                    "omega": new_omega,
                    "opacity": new_opacity,
                    "features_dc": new_features_dc,
                }

                optimizable_tensors = self._extend_param_group(param_dict)

                self._xyz = optimizable_tensors["xyz"]
                self._t = optimizable_tensors["t"]
                self._xyz_scales = optimizable_tensors["xyz_scales"]
                self._t_scale = optimizable_tensors["t_scale"]
                self._rotation = optimizable_tensors["rotation"]
                self._motion = optimizable_tensors["motion"]
                self._omega = optimizable_tensors["omega"]
                self._opacity = optimizable_tensors["opacity"]
                self._features_dc = optimizable_tensors["features_dc"]

                self._max_vspace_gradient = torch.zeros(
                    (self._xyz.shape[0], 1), device=self._device
                )
                self._density_control_denom = torch.zeros(
                    (self._xyz.shape[0], 1), device=self._device
                )
                self._max_radii2D = torch.zeros(
                    (self._xyz.shape[0]), device=self._device
                )

                self._prune_points(
                    torch.cat(
                        (
                            space_mask,
                            torch.zeros(
                                self._split_num * space_mask.sum(),
                                device=self._device,
                                dtype=bool,
                            ),
                        )
                    )
                )

    def _prune_points(self, mask: torch.Tensor):
        mask = mask.squeeze()

        valid_point_mask = torch.logical_not(mask)

        param_list = [
            "xyz",
            "t",
            "xyz_scales",
            "t_scale",
            "rotation",
            "motion",
            "omega",
            "opacity",
            "features_dc",
        ]
        param_dict = {param: valid_point_mask for param in param_list}

        gaussian_optimizable_tensors = self._prune_param_group(param_dict)
        self._xyz = gaussian_optimizable_tensors["xyz"]
        self._t = gaussian_optimizable_tensors["t"]
        self._xyz_scales = gaussian_optimizable_tensors["xyz_scales"]
        self._t_scale = gaussian_optimizable_tensors["t_scale"]
        self._rotation = gaussian_optimizable_tensors["rotation"]
        self._motion = gaussian_optimizable_tensors["motion"]
        self._omega = gaussian_optimizable_tensors["omega"]
        self._opacity = gaussian_optimizable_tensors["opacity"]
        self._features_dc = gaussian_optimizable_tensors["features_dc"]

        self._max_vspace_gradient = self._max_vspace_gradient[valid_point_mask]
        self._density_control_denom = self._density_control_denom[valid_point_mask]
        self._max_radii2D = self._max_radii2D[valid_point_mask]
        torch.cuda.empty_cache()

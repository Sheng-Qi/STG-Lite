import os
import torch
import torch.nn as nn
import numpy as np
from typing import Type
import logging
import math
from plyfile import PlyData, PlyElement
from pydantic import BaseModel, Field

from scene.dataset.abstract_dataset import AbstractDataset
from scene.model.basic_gaussian_model import BasicGaussianModel
from scene.cameras import Camera
from utils.math_utils import trbfunction
from utils.general_utils import build_rotation, inverse_sigmoid
from utils.graphics_utils import BasicPointCloud
from utils.optimizer_utils import extend_param_group, prune_param_group


class TimeDensityControlParams(BaseModel):
    enable: bool
    split_ratio: float
    th_grad: float
    th_scale: float = Field(..., gt=0.0)


class SpacetimeLearningRateParams(BaseModel):
    t: float
    t_scale: float
    motion: float
    omega: float


class SpacetimeGaussianModelParams(BaseModel):
    t_scale_init: float
    motion_degree: int = Field(1, ge=1, le=3)
    omega_degree: int = Field(1, ge=0, le=1)
    time_density_control: TimeDensityControlParams
    learning_rate: SpacetimeLearningRateParams


class SpacetimeGaussianModel(BasicGaussianModel):

    def __init__(self, params: dict, context: dict):
        spacetime_params = params["spacetime"]
        non_spacetime_params = {k: v for k, v in params.items() if k != "spacetime"}
        super().__init__(non_spacetime_params, context)
        self._spacetime_params = SpacetimeGaussianModelParams.model_validate(
            spacetime_params
        )

        self._FULL_MOTION_DEGREE = 3
        self._FULL_OMEGA_DEGREE = 1

        self._max_vtime_gradient: torch.Tensor = None
        self._max_t_scale_active: torch.Tensor = None

        self.__t: torch.Tensor = None
        self.__t_scale: torch.Tensor = None
        self.__motion: torch.Tensor = None
        self.__omega: torch.Tensor = None

    @property
    def t(self) -> torch.Tensor:
        return self.__t

    @property
    def t_scale(self) -> torch.Tensor:
        return self.__t_scale

    @property
    def t_scale_active(self) -> torch.Tensor:
        return torch.exp(self.__t_scale)

    @property
    def motion(self) -> torch.Tensor:
        return self.__motion

    @property
    def motion_full_degree(self) -> torch.Tensor:
        return torch.cat(
            [
                self.__motion,
                torch.zeros(
                    (
                        self.__motion.shape[0],
                        3
                        * (
                            self._FULL_MOTION_DEGREE
                            - self._spacetime_params.motion_degree
                        ),
                    ),
                    device=self._device,
                ),
            ],
            dim=1,
        )

    @property
    def omega(self) -> torch.Tensor:
        return self.__omega

    @property
    def omega_full_degree(self) -> torch.Tensor:
        return torch.cat(
            [
                self.__omega,
                torch.zeros(
                    (
                        self.__omega.shape[0],
                        4
                        * (
                            self._FULL_OMEGA_DEGREE
                            - self._spacetime_params.omega_degree
                        ),
                    ),
                    device=self._device,
                ),
            ],
            dim=1,
        )

    def xyz_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        output = self.xyz.clone()
        for degree in range(1, self._spacetime_params.motion_degree + 1):
            output = (
                output + self.motion[:, (degree - 1) * 3 : degree * 3] * delta_t**degree
            )
        return output

    def rotation_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        output = self.rotation.clone()
        for degree in range(1, self._spacetime_params.omega_degree + 1):
            output = (
                output + self.omega[:, (degree - 1) * 4 : degree * 4] * delta_t**degree
            )
        return output

    def rotation_activated_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.rotation_projected(delta_t))

    def opacity_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(self.opacity_activated_projected(delta_t))

    def opacity_activated_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return self.opacity_activated * trbfunction(delta_t / self.t_scale_active)

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        self._vspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype, requires_grad=True, device=self.device
        )
        self._vspace_points.retain_grad()
        raster_settings = GRsetting(
            image_height=int(camera.resized_height),
            image_width=int(camera.resized_width),
            tanfovx=math.tan(camera.camera_info.FovX * 0.5),
            tanfovy=math.tan(camera.camera_info.FovY * 0.5),
            bg=self._bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self._active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            antialiasing=False,
            debug=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)

        delta_t = (
            torch.full(
                (self.xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=self.xyz.dtype,
                requires_grad=False,
                device=self.device,
            )
            - self.t
        )

        result = rasterizer(
            means3D=self.xyz_projected(delta_t),
            means2D=self._vspace_points,
            shs=self.features,
            colors_precomp=None,
            opacities=self.opacity_activated_projected(delta_t),
            scales=self.xyz_scales_activated,
            rotations=self.rotation_activated_projected(delta_t),
            cov3D_precomp=None,
        )

        assert len(result) == 3 + int(self._context.is_render_support_vspace), "Invalid result length"
        if len(result) == 4:
            self._rendered_image, self._vspace_radii, self._rendered_depth, self._vspace_values = result
        elif len(result) == 3:
            self._rendered_image, self._vspace_radii, self._rendered_depth = result
        else:
            raise ValueError("Invalid result length")

        if self._basic_params.color_transform.enable:
            self._rendered_image = self._apply_color_transformation(
                camera, self._rendered_image
            )

        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def render_forward_only(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        screenspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype, requires_grad=True, device=self.device
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
            bg=self._bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self._active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)
        delta_t = (
            torch.full(
                (self.xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=self.xyz.dtype,
                requires_grad=False,
                device=self.device,
            )
            - self.t
        )
        rendered_image, radii = rasterizer(
            timestamp=camera.camera_info.timestamp_ratio,
            trbfcenter=self.t,
            trbfscale=self.t_scale_active,
            motion=self.motion_full_degree,
            means3D=self.xyz,
            means2D=screenspace_points,
            shs=self.features,
            colors_precomp=None,
            opacities=self.opacity_activated,
            scales=self.xyz_scales_activated,
            rotations=self.rotation_activated_projected(delta_t),
            cov3D_precomp=None,
        )

        if self._basic_params.color_transform.enable:
            rendered_image = self._apply_color_transformation_forward_only(
                camera, rendered_image
            )

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        return {
            "rendered_image": rendered_image,
            "radii": radii,
            "duration": duration,
        }

    def _set_gaussian_attributes_time(self, param_dict: dict):
        assert all(
            key in param_dict
            for key in [
                "t",
                "t_scale",
                "motion",
                "omega",
            ]
        ), "Missing required parameters in param_dict"
        assert all(
            param_dict["t"].shape[0] == param_dict[key].shape[0]
            for key in [
                "t_scale",
                "motion",
                "omega",
            ]
        ), "Parameter shapes do not match"
        self.__t = param_dict["t"]
        self.__t_scale = param_dict["t_scale"]
        self.__motion = param_dict["motion"]
        self.__omega = param_dict["omega"]

    def _set_gaussian_attributes_all(self, param_dict):
        assert (
            param_dict["xyz"].shape[0] == param_dict["t"].shape[0]
        ), "Parameter shapes do not match"
        self._set_gaussian_attributes(param_dict)
        self._set_gaussian_attributes_time(param_dict)

    def _init_point_cloud_parameters(self, dataset: AbstractDataset):
        mask = super()._init_point_cloud_parameters(dataset)
        times = torch.tensor(np.asarray(dataset.ply_data.times)).float().to(self._device)
        times = times[mask]
        param_dict = {
            "t": nn.Parameter(times.contiguous().requires_grad_(True)),
            "t_scale": nn.Parameter(
                torch.full(
                    (times.shape[0], 1),
                    self._spacetime_params.t_scale_init,
                    device=self._device,
                    requires_grad=True,
                )
            ),
            "motion": nn.Parameter(
                torch.zeros(
                    (times.shape[0], self._spacetime_params.motion_degree * 3),
                    device=self._device,
                ).requires_grad_(True)
            ),
            "omega": nn.Parameter(
                torch.zeros(
                    (times.shape[0], self._spacetime_params.omega_degree * 4),
                    device=self._device,
                ).requires_grad_(True)
            ),
        }
        self._set_gaussian_attributes_time(param_dict)
        return mask

    def _fetch_point_cloud_parameters(self, ply_data: PlyData):
        super()._fetch_point_cloud_parameters(ply_data)

        t_names = ["trbf_center"]
        t_scale_names = ["trbf_scale"]
        motion_names = [
            "motion_" + str(i) for i in range(self._spacetime_params.motion_degree * 3)
        ]
        omega_names = [
            "omega_" + str(i) for i in range(self._spacetime_params.omega_degree * 4)
        ]
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

        _param_dict = {
            "t": nn.Parameter(
                torch.tensor(t, dtype=torch.float, device=self._device).requires_grad_(
                    True
                )
            ),
            "t_scale": nn.Parameter(
                torch.tensor(
                    t_scale, dtype=torch.float, device=self._device
                ).requires_grad_(True)
            ),
            "motion": nn.Parameter(
                torch.tensor(
                    motion, dtype=torch.float, device=self._device
                ).requires_grad_(True)
            ),
            "omega": nn.Parameter(
                torch.tensor(
                    omega, dtype=torch.float, device=self._device
                ).requires_grad_(True)
            ),
        }
        self._set_gaussian_attributes_time(_param_dict)

    def _save_point_cloud_parameters(self, pcd_path):
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        trbf_center = self.t.detach().cpu().numpy()
        trbf_scale = self.t_scale.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        motion_full_degree = self.motion_full_degree.detach().cpu().numpy()
        f_dc = (
            self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        )
        f_rest = (
            self.features_rest.transpose(1, 2)
            .flatten(start_dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.xyz_scales.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()
        omega_full_degree = self.omega_full_degree.detach().cpu().numpy()

        list_of_attributes = []
        list_of_attributes.extend(["x", "y", "z"])
        list_of_attributes.extend(["trbf_center"])
        list_of_attributes.extend(["trbf_scale"])
        list_of_attributes.extend(["nx", "ny", "nz"])
        list_of_attributes.extend(["motion_" + str(i) for i in range(9)])
        list_of_attributes.extend(["f_dc_" + str(i) for i in range(3)])
        list_of_attributes.extend(
            [
                "f_rest_" + str(i)
                for i in range(3 * (self._basic_params.sh_degree + 1) ** 2 - 3)
            ]
        )
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
                motion_full_degree,
                f_dc,
                f_rest,
                opacity,
                scale,
                rotation,
                omega_full_degree,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        pcd_vertex_element = PlyElement.describe(elements, "vertex")
        PlyData([pcd_vertex_element]).write(pcd_path)

    @property
    def _initial_param_groups(self) -> list:
        time_param_groups = [
            {
                "params": [self.__t],
                "lr": self._spacetime_params.learning_rate.t,
                "name": "t",
            },
            {
                "params": [self.__t_scale],
                "lr": self._spacetime_params.learning_rate.t_scale,
                "name": "t_scale",
            },
            {
                "params": [self.__motion],
                "lr": self._basic_params.learning_rate.xyz.init
                * self._context.camera_extent
                * 0.5
                * self._spacetime_params.learning_rate.motion,
                "name": "motion",
            },
            {
                "params": [self.__omega],
                "lr": self._spacetime_params.learning_rate.omega,
                "name": "omega",
            },
        ]
        return super()._initial_param_groups + time_param_groups

    def _init_density_control(self):
        super()._init_density_control()
        if self._spacetime_params.time_density_control.enable:
            self._max_vtime_gradient = torch.zeros(
                (self.xyz.shape[0], 1), device=self._device
            )
            self._max_t_scale_active = torch.zeros(
                (self.xyz.shape[0], 1), device=self._device
            )

    def _cache_density_control(self):
        super()._cache_density_control()
        if self._spacetime_params.time_density_control.enable:
            self._max_vtime_gradient[self._radii > 0] = torch.max(
                self._max_vtime_gradient[self._radii > 0],
                torch.norm(self.t.grad[self._radii > 0], dim=-1, keepdim=True),
            )
            self._max_t_scale_active[self._radii > 0] = torch.max(
                self._max_t_scale_active[self._radii > 0],
                torch.exp(self.t_scale[self._radii > 0]),
            )

    def _density_control(
        self, iteration: int, camera: Camera, dataset: AbstractDataset
    ):
        if iteration == self._basic_params.density_control.start:
            self._init_density_control()
        if iteration in range(
            self._basic_params.density_control.start,
            self._basic_params.density_control.end,
        ):
            self._cache_density_control()
        if iteration in range(
            self._basic_params.density_control.start
            + self._basic_params.density_control.step,
            self._basic_params.density_control.end,
            self._basic_params.density_control.step,
        ):
            split_num = self._basic_params.density_control.split_num
            if self._basic_params.density_control.th_method == "complex":
                space_mask = (
                    self._max_vspace_grads.squeeze()
                    * self._max_vspace_radii.squeeze()
                    * torch.sqrt(self.opacity_activated.squeeze())
                    > self._basic_params.density_control.grad_th_xyz
                )
                space_mask = torch.logical_and(
                    space_mask, self.opacity_activated.squeeze() > 0.15
                )
            elif self._basic_params.density_control.th_method == "simple":
                space_mask = (
                    self._max_vspace_grads.squeeze()
                    > self._basic_params.density_control.grad_th_xyz
                )
            else:
                raise ValueError("Invalid threshold method")

            if self._spacetime_params.time_density_control.enable:
                time_mask = (
                    self._max_vtime_gradient.squeeze()
                    * torch.sqrt(torch.sigmoid(self.opacity).squeeze())
                    / self._max_vspace_radii.squeeze()
                    / torch.max(
                        torch.exp(self.t_scale),
                        torch.full_like(
                            self.t_scale,
                            self._spacetime_params.time_density_control.th_scale,
                            device=self._device,
                        ),
                    ).squeeze()
                    > self._spacetime_params.time_density_control.th_grad
                )
                time_mask[space_mask] = False
            else:
                time_mask = torch.zeros_like(space_mask, device=self._device)

            selected_mask = torch.logical_or(space_mask, time_mask)

            logging.debug(
                f"Space mask: {space_mask.sum()}, Time mask: {time_mask.sum()}"
            )

            space_split_mask = torch.logical_and(
                space_mask,
                torch.max(torch.exp(self.xyz_scales), dim=1).values
                > self._basic_params.density_control.scale_th_xyz
                * self._context.camera_extent,
            )

            space_split_mask_under_selection = torch.logical_and(
                space_split_mask, selected_mask
            )[selected_mask].repeat(split_num)

            time_split_mask = torch.logical_and(
                time_mask,
                torch.exp(self.t_scale).squeeze()
                > self._spacetime_params.time_density_control.th_scale,
            )
            time_split_mask_under_selection = torch.logical_and(
                time_split_mask, selected_mask
            )[selected_mask].repeat(split_num)

            new_xyz = self.xyz[selected_mask].repeat(split_num, 1)
            new_t = self.t[selected_mask].repeat(split_num, 1)
            new_xyz_scales = self.xyz_scales[selected_mask].repeat(split_num, 1)
            new_t_scale = self.t_scale[selected_mask].repeat(split_num, 1)
            new_rotation = self.rotation[selected_mask].repeat(split_num, 1)
            new_motion = self.motion[selected_mask].repeat(split_num, 1)
            new_omega = self.omega[selected_mask].repeat(split_num, 1)
            new_opacity = self.opacity[selected_mask].repeat(split_num, 1)
            new_features_dc = self.features_dc[selected_mask].repeat(split_num, 1, 1)
            new_features_rest = self.features_rest[selected_mask].repeat(
                split_num, 1, 1
            )

            stds = torch.exp(self.xyz_scales[selected_mask]).repeat(split_num, 1)
            means = torch.zeros((stds.size(0), 3), device=self._device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self.rotation[selected_mask]).repeat(split_num, 1, 1)
            new_xyz_random_offset = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_xyz[space_split_mask_under_selection] += new_xyz_random_offset[
                space_split_mask_under_selection
            ]

            new_t_offset = torch.abs(
                torch.randn_like(new_t) * torch.exp(new_t_scale) / 2
            )
            assert hasattr(dataset, "duration"), "Dataset must have duration"
            aligned_t = (
                torch.floor((new_t + new_t_offset) * dataset.duration)
                / dataset.duration
            )
            new_t[time_split_mask_under_selection] = aligned_t[
                time_split_mask_under_selection
            ]
            new_t = torch.clamp(new_t, 0.0, 1.0)
            delta_t = new_t - self.t[selected_mask].repeat(split_num, 1)

            for degree in range(1, self._spacetime_params.motion_degree + 1):
                new_xyz += (
                    new_motion[:, (degree - 1) * 3 : degree * 3] * delta_t**degree
                )
                if degree != self._spacetime_params.motion_degree:
                    new_motion[:, (degree - 1) * 3 : degree * 3] += (
                        new_motion[:, degree * 3 : (degree + 1) * 3] * delta_t
                    )

            for degree in range(1, self._spacetime_params.omega_degree + 1):
                new_rotation += (
                    new_omega[:, (degree - 1) * 4 : degree * 4] * delta_t**degree
                )
                if degree != self._spacetime_params.omega_degree:
                    new_omega[:, (degree - 1) * 4 : degree * 4] += (
                        new_omega[:, degree * 4 : (degree + 1) * 4] * delta_t
                    )

            new_xyz_scales[space_split_mask_under_selection] = torch.log(
                torch.exp(new_xyz_scales[space_split_mask_under_selection])
                * self._basic_params.density_control.split_ratio
            )
            new_t_scale[time_split_mask_under_selection] = torch.log(
                torch.exp(new_t_scale[time_split_mask_under_selection])
                * self._spacetime_params.time_density_control.split_ratio
            )

            self._extend_spacetime_parameters(
                new_xyz,
                new_t,
                new_xyz_scales,
                new_t_scale,
                new_rotation,
                new_motion,
                new_omega,
                new_opacity,
                new_features_dc,
                new_features_rest,
            )

            self._init_density_control()

            self._prune_mask(
                torch.cat(
                    (
                        selected_mask,
                        torch.zeros(
                            split_num * selected_mask.sum(),
                            device=self._device,
                            dtype=bool,
                        ),
                    )
                )
            )

    def _extend_spacetime_parameters(
        self,
        new_xyz,
        new_t,
        new_xyz_scales,
        new_t_scale,
        new_rotation,
        new_motion,
        new_omega,
        new_opacity,
        new_features_dc,
        new_features_rest,
    ):
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
            "features_rest": new_features_rest,
        }

        optimizable_tensors = extend_param_group(self.optimizer, param_dict)

        self._set_gaussian_attributes_all(optimizable_tensors)

    def _prune_mask(self, mask: torch.Tensor):
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
            "features_rest",
        ]
        param_dict = {param: valid_point_mask for param in param_list}

        gaussian_optimizable_tensors = prune_param_group(self.optimizer, param_dict)
        self._set_gaussian_attributes_all(gaussian_optimizable_tensors)
        if (
            self._max_vspace_radii is not None
            and self._max_vspace_radii.shape[0] == mask.shape[0]
        ):
            self._max_vspace_grads = self._max_vspace_grads[valid_point_mask]
            self._densification_denom = self._densification_denom[valid_point_mask]
            self._max_vspace_radii = self._max_vspace_radii[valid_point_mask]
        if (
            self._vspace_radii is not None
            and self._vspace_radii.shape[0] == mask.shape[0]
        ):
            self._vspace_radii = self._vspace_radii[valid_point_mask]
            self._vspace_points = self._vspace_points[valid_point_mask]
        if (
            self._spacetime_params.time_density_control.enable
            and self._max_vtime_gradient is not None
            and self._max_vtime_gradient.shape[0] == mask.shape[0]
        ):
            self._max_vtime_gradient = self._max_vtime_gradient[valid_point_mask]
            self._max_t_scale_active = self._max_t_scale_active[valid_point_mask]
        torch.cuda.empty_cache()

    def _calculate_mask_penalty_loss(self, camera: Camera):
        if not self._context.is_render_support_vspace:
            self._vspace_values = camera.project_points(self.xyz)

        vspace_values = self._vspace_values
        N, _ = vspace_values.shape
        penalty = torch.zeros(N, device=self.device)
        W, H = camera.resized_width, camera.resized_height
        in_bound_mask = (
            (vspace_values[:, 0] >= 0)
            & (vspace_values[:, 0] < W)
            & (vspace_values[:, 1] >= 0)
            & (vspace_values[:, 1] < H)
        )
        non_zero_mask = (vspace_values[:, 0] != 0) & (vspace_values[:, 1] != 0)
        opacity_mask = (
            self.opacity_activated_projected(camera.camera_info.timestamp_ratio)
            > self._basic_params.prune_points.th_opacity
        ).squeeze()

        in_view_mask = torch.logical_and(
            in_bound_mask, torch.logical_and(non_zero_mask, opacity_mask)
        )
        if in_view_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        x_int = vspace_values[in_view_mask, 0].long()
        y_int = vspace_values[in_view_mask, 1].long()

        distances = camera.mask_distance_map[y_int, x_int].float()
        penalty[in_view_mask] = distances / torch.max(
            torch.tensor([W, H], device=self.device)
        )

        return self._basic_params.lambda_mask * torch.mean(penalty)

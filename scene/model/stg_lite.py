import os
import torch
import torch.nn as nn
import numpy as np
from typing import Type
import logging
import math
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from scene.model.abstract_gaussian_model import AbstractModel
from scene.cameras import Camera
from utils.math_utils import trbfunction
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import color_transform_regularization_loss


class GaussianModel(AbstractModel):

    def __init__(self, model_params: dict):
        # Model parameters
        super().__init__(model_params)
        self._cameras_extent = model_params["cameras_extent"]
        self._enable_color_transform: bool = model_params["enable_color_transform"]
        self._lambda_color_transform: float = model_params["lambda_color_transform"]
        self._color_transform_matrix_model_path_prior: str = model_params[
            "color_transform_matrix_model_path_prior"
        ]
        self._t_scale_init: float = model_params["t_scale_init"]
        self._split_num: int = model_params["split_num"]
        self._split_ratio: float = model_params["split_ratio"]
        self._grad_threshold_xyz: float = model_params["grad_threshold_xyz"]
        self._percent_dense_xyz: float = model_params["percent_dense_xyz"]
        self._densification_start: int = model_params["densification_start"]
        self._densification_step: int = model_params["densification_step"]
        self._densification_end: int = model_params["densification_end"]
        self._reset_opacity_start: int = model_params["reset_opacity_start"]
        self._reset_opacity_step: int = model_params["reset_opacity_step"]
        self._reset_opacity_end: int = model_params["reset_opacity_end"]
        self._prune_points_start: int = model_params["prune_points_start"]
        self._prune_points_step: int = model_params["prune_points_step"]
        self._prune_points_end: int = model_params["prune_points_end"]
        self._prune_threshold: float = model_params["prune_threshold"]
        self._remove_outlier_iterations: list[int] = model_params[
            "remove_outlier_iterations"
        ]
        self._learning_rate: dict = model_params["learning_rate"]
        self._white_background: bool = model_params["white_background"]

        # Other parameters
        self._background_color = torch.tensor(
            [1, 1, 1] if self._white_background else [0 for i in range(9)],
            dtype=torch.float32,
            device=self._device,
        )
        self._max_bounds_init = None
        self._min_bounds_init = None
        self._xyz_scheduler_args = None
        self._color_transformation_scheduler_args = None
        self._optimizer = None

        # Color transform parameters
        self._color_transformation_matrix_dict = {}

        # Cached renderings
        self._rendered_image = None
        self._radii = None
        self._rendered_depth = None
        self._screenspace_points = None

        # N-dimensional tensors
        self._max_vspace_gradient = None
        self._density_control_denom = None
        self._max_radii2D = None

        # N-dimensional optimizable tensors
        self._xyz = None
        self._t = None
        self._xyz_scales = None
        self._t_scale = None
        self._rotation = None
        self._motion = None
        self._omega = None
        self._opacity = None
        self._features_dc = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def init(self, pcd_data: BasicPointCloud):
        point_cloud = torch.tensor(np.asarray(pcd_data.points)).float().to(self._device)
        color = torch.tensor(np.asarray(pcd_data.colors)).float().to(self._device)
        times = torch.tensor(np.asarray(pcd_data.times)).float().to(self._device)

        dist2 = torch.clamp_min(
            distCUDA2(
                torch.from_numpy(np.asarray(pcd_data.points)).float().to(self._device)
            ),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((point_cloud.shape[0], 4), device=self._device)
        rots[:, 0] = 1

        opactities = inverse_sigmoid(
            0.1
            * torch.ones(
                (point_cloud.shape[0], 1), dtype=torch.float, device=self._device
            )
        )

        self._xyz = nn.Parameter(point_cloud.contiguous().requires_grad_(True))
        self._t = nn.Parameter(times.contiguous().requires_grad_(True))
        self._xyz_scales = nn.Parameter(scales.requires_grad_(True))
        self._t_scale = nn.Parameter(
            torch.full(
                (point_cloud.shape[0], 1),
                self._t_scale_init,
                device=self._device,
                requires_grad=True,
            )
        )
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._motion = nn.Parameter(
            torch.zeros((point_cloud.shape[0], 9), device=self._device).requires_grad_(
                True
            )
        )
        self._omega = nn.Parameter(
            torch.zeros((point_cloud.shape[0], 4), device=self._device).requires_grad_(
                True
            )
        )
        self._opacity = nn.Parameter(opactities.requires_grad_(True))
        self._features_dc = nn.Parameter(color.contiguous().requires_grad_(True))

        self._max_bounds_init = [torch.amax(point_cloud[:, i]) for i in range(3)]
        self._min_bounds_init = [torch.amin(point_cloud[:, i]) for i in range(3)]

        if self._enable_color_transform and self._color_transform_matrix_model_path_prior is not None:
            if os.path.exists(self._color_transform_matrix_model_path_prior):
                self._color_transformation_matrix_dict = torch.load(
                    self._color_transform_matrix_model_path_prior
                )
                for key, matrix_param in self._color_transformation_matrix_dict.items():
                    matrix_param.to(self._device)
            else:
                logging.warning(
                    f"Color transformation matrix model not found at {self._color_transform_matrix_model_path_prior}"
                )

        self._setup()

    def load(
        self,
        pcd_path: str,
        init_pcd_data: BasicPointCloud
    ):
        plydata = PlyData.read(pcd_path)

        xyz_names = ["x", "y", "z"]
        t_names = ["trbf_center"]
        xyz_scales_names = ["scale_" + str(i) for i in range(3)]
        t_scale_names = ["trbf_scale"]
        rotation_names = ["rot_" + str(i) for i in range(4)]
        motion_names = ["motion_" + str(i) for i in range(9)]
        omega_names = ["omega_" + str(i) for i in range(4)]
        opacity_names = ["opacity"]
        features_dc_names = ["f_dc_" + str(i) for i in range(3)]

        xyz = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in xyz_names], axis=1
        )
        t = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in t_names], axis=1
        )
        xyz_scales = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in xyz_scales_names], axis=1
        )
        t_scale = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in t_scale_names], axis=1
        )
        rotation = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in rotation_names], axis=1
        )
        motion = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in motion_names], axis=1
        )
        omega = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in omega_names], axis=1
        )
        opacity = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in opacity_names], axis=1
        )
        features_dc = np.stack(
            [np.asarray(plydata.elements[0][name]) for name in features_dc_names],
            axis=1,
        )

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self._device).requires_grad_(
                True
            )
        )
        self._t = nn.Parameter(
            torch.tensor(t, dtype=torch.float, device=self._device).requires_grad_(True)
        )
        self._xyz_scales = nn.Parameter(
            torch.tensor(
                xyz_scales, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._t_scale = nn.Parameter(
            torch.tensor(
                t_scale, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                rotation, dtype=torch.float, device=self._device
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
        self._opacity = nn.Parameter(
            torch.tensor(
                opacity, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(
                features_dc, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )

        point_cloud = (
            torch.tensor(np.asarray(init_pcd_data.points)).float().to(self._device)
        )
        self._max_bounds_init = [torch.amax(point_cloud[:, i]) for i in range(3)]
        self._min_bounds_init = [torch.amin(point_cloud[:, i]) for i in range(3)]

        if self._enable_color_transform:
            if self._color_transform_matrix_model_path_prior is None:
                self._color_transform_matrix_model_path_prior = pcd_path.replace(
                    ".ply", "_color_transform_matrix.pth"
                )
            if os.path.exists(self._color_transform_matrix_model_path_prior):
                self._color_transformation_matrix_dict = torch.load(
                    self._color_transform_matrix_model_path_prior
                )
                for key, matrix_param in self._color_transformation_matrix_dict.items():
                    matrix_param.to(self._device)
            else:
                logging.warning(
                    f"Color transformation matrix model not found at {self._color_transform_matrix_model_path_prior}"
                )

        self._setup()

    def save(self, pcd_path: str):
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

        if self._enable_color_transform:
            self._color_transform_matrix_model_path_prior = pcd_path.replace(
                ".ply", "_color_transform_matrix.pth"
            )
            torch.save(
                self._color_transformation_matrix_dict,
                self._color_transform_matrix_model_path_prior,
            )

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
            means3D=self._get_xyz_projected(delta_t),
            means2D=self._screenspace_points,
            shs=None,
            colors_precomp=self._features_dc,
            opacities=self._get_opacity_projected(delta_t),
            scales=torch.exp(self._xyz_scales),
            rotations=self._get_rotation_projected(delta_t),
            cov3D_precomp=None,
        )

        if self._enable_color_transform:
            camera_key = f"color_transformation_matrix_{camera.camera_info.camera_id}"
            if camera_key not in self._color_transformation_matrix_dict:
                matrix_param = nn.Parameter(
                    torch.tensor(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                        dtype=torch.float32,
                        device=self._device,
                    ),
                    requires_grad=True,
                )
                new_param_group = {
                    "params": [matrix_param],
                    "lr": self._learning_rate["color_transformation_init"],
                    "name": camera_key,
                }
                self._optimizer.add_param_group(new_param_group)
                self._color_transformation_matrix_dict[camera_key] = matrix_param
            else:
                matrix_param = self._color_transformation_matrix_dict[camera_key]

            image_stack_with_alpha = torch.cat(
                [self._rendered_image, torch.ones_like(self._rendered_image[:1])], dim=0
            )
            self._rendered_image = torch.einsum(
                "ij, jwh -> iwh", matrix_param, image_stack_with_alpha
            )

        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def render_forward(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
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
            camera_key = f"color_transformation_matrix_{camera.camera_info.camera_id}"
            if camera_key in self._color_transformation_matrix_dict:
                matrix_param = self._color_transformation_matrix_dict[camera_key]
                image_stack_with_alpha = torch.cat(
                    [rendered_image, torch.ones_like(rendered_image[:1])], dim=0
                )
                rendered_image = torch.einsum(
                    "ij, jwh -> iwh", matrix_param, image_stack_with_alpha
                )

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        return {
            "rendered_image": rendered_image,
            "radii": radii,
            "duration": duration,
        }

    def iteration_start(self, iteration: int, camera: Camera):
        self._density_control(iteration)
        if iteration in range(
            self._reset_opacity_start, self._reset_opacity_end, self._reset_opacity_step
        ):
            self._reset_opacity()
        if iteration in self._remove_outlier_iterations:
            self._remove_outlier()
        if iteration in range(
            self._prune_points_start, self._prune_points_end, self._prune_points_step
        ):
            self._prune_points(torch.sigmoid(self._opacity) < self._prune_threshold)

        self._update_learning_rate(iteration)

    def get_regularization_loss(self, camera: Camera) -> torch.Tensor:
        loss = super().get_regularization_loss(camera)
        if self._enable_color_transform:
            matrix_param = self._color_transformation_matrix_dict[
                f"color_transformation_matrix_{camera.camera_info.camera_id}"
            ]
            default_matrix = torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                dtype=torch.float32,
                device=self._device,
            )
            loss += color_transform_regularization_loss(
                matrix_param, default_matrix, torch.tensor(self._lambda_color_transform, device=self._device)
            )
        return loss

    def __len__(self) -> int:
        return self._xyz.shape[0] if self._xyz is not None else 0

    def _setup(self):
        self._max_vspace_gradient = torch.zeros(
            (self._xyz.shape[0], 1), device=self._device
        )
        self._density_control_denom = torch.zeros(
            (self._xyz.shape[0], 1), device=self._device
        )
        self._max_radii2D = torch.zeros((self._xyz.shape[0]), device=self._device)
        self._initialize_learning_rate()

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

    def _update_learning_rate(self, iteration: int):
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self._xyz_scheduler_args(iteration)
            elif param_group["name"].startswith("color_transformation_matrix"):
                param_group["lr"] = self._color_transformation_scheduler_args(iteration)

    def _density_control(self, iteration: int):
        if iteration in range(self._densification_start, self._densification_end):
            if self._radii is None:
                raise ValueError(
                    "Radii not computed, please set densification_start to a positive value"
                )
            self._max_radii2D[self._radii > 0] = torch.max(
                self._max_radii2D[self._radii > 0], self._radii[self._radii > 0]
            )
            self._max_vspace_gradient[self._radii > 0] = torch.max(
                self._max_vspace_gradient[self._radii > 0],
                torch.norm(
                    self._screenspace_points.grad[self._radii > 0, :2], dim=-1, keepdim=True
                ),
            )
            self._density_control_denom[self._radii > 0] += 1
            if (iteration + 1) in range(
                self._densification_start,
                self._densification_end,
                self._densification_step,
            ):
                space_mask = (
                    self._max_vspace_gradient.squeeze()
                    * torch.pow(
                        self._density_control_denom.squeeze() / self._densification_step, 2
                    )
                    * self._max_radii2D.squeeze()
                    * torch.sqrt(torch.sigmoid(self._opacity).squeeze())
                    > self._grad_threshold_xyz
                )
                space_mask = torch.logical_and(
                    space_mask,
                    torch.sigmoid(self._opacity).squeeze() > 0.15
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

    def _reset_opacity(self):
        new_opacity = inverse_sigmoid(
            0.1
            * torch.ones(
                (self._xyz.shape[0], 1), dtype=torch.float, device=self._device
            )
        )
        self._reset_param_group({"opacity": new_opacity})

    def _remove_outlier(self):
        max_bounds = torch.tensor(self._max_bounds_init, device=self._device)
        min_bounds = torch.tensor(self._min_bounds_init, device=self._device)
        mask_max = torch.any(self._xyz > max_bounds, dim=1)
        mask_min = torch.any(self._xyz < min_bounds, dim=1)
        mask = torch.logical_or(mask_max, mask_min)
        self._prune_points(mask)

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

    def _add_param_group(self, param_dict: dict):
        for key, param_group in param_dict.items():
            self._optimizer.add_param_group(param_group)

    def _reset_param_group(self, param_dict: dict):
        for group in self._optimizer.param_groups:
            if group["name"] in param_dict:
                reset_value = param_dict[group["name"]]
                stored_state = self._optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(reset_value)
                stored_state["exp_avg_sq"] = torch.zeros_like(reset_value)
                del self._optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(reset_value.requires_grad_(True))
                self._optimizer.state[group["params"][0]] = stored_state
                self._opacity = group["params"][0]

    def _prune_param_group(self, param_dict: dict) -> dict:
        gaussian_optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            if group["name"] in param_dict:
                mask = param_dict[group["name"]]
                stored_state = self._optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self._optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                    self._optimizer.state[group["params"][0]] = stored_state
                    gaussian_optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].requires_grad_(True)
                    )
                    gaussian_optimizable_tensors[group["name"]] = group["params"][0]
        return gaussian_optimizable_tensors

    def _extend_param_group(self, param_dict: dict) -> dict:
        optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            if group["name"] in param_dict:
                extension_tensor = param_dict[group["name"]]
                stored_state = self._optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self._optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self._optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _get_xyz_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return (
            self._xyz
            + self._motion[:, :3] * delta_t
            + self._motion[:, 3:6] * delta_t**2
            + self._motion[:, 6:] * delta_t**3
        )

    def _get_rotation_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation + delta_t * self._omega)

    def _get_opacity_projected(self, delta_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._opacity) * trbfunction(
            delta_t / torch.exp(self._t_scale)
        )

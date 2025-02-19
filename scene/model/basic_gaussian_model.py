import os
import torch
import torch.nn as nn
import numpy as np
from typing import Type
import logging
import math
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from scene.dataset.abstract_dataset import AbstractDataset
from scene.model.abstract_gaussian_model import AbstractModel
from scene.cameras import Camera
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import matrix_loss


class BasicGaussianModel(AbstractModel):

    def __init__(self, model_params: dict):
        # Model parameters
        super().__init__(model_params)
        self._cameras_extent = model_params["cameras_extent"]
        self._enable_color_transform: bool = model_params["enable_color_transform"]
        self._lambda_color_transform: float = model_params["lambda_color_transform"]
        self._color_transform_matrix_model_path_prior: str = model_params[
            "color_transform_matrix_model_path_prior"
        ]
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

        # N-dimensional optimizable tensors
        self._xyz: torch.Tensor = None
        self._xyz_scales: torch.Tensor = None
        self._rotation: torch.Tensor = None
        self._opacity: torch.Tensor = None
        self._features_dc: torch.Tensor = None

        # Color transform parameters
        self._color_transformation_matrix_dict = {}

        # Cached rendering results
        self._rendered_image = None
        self._radii = None
        self._rendered_depth = None
        self._screenspace_points = None

        # Cached densification parameters
        self._max_vspace_gradient = None
        self._density_control_denom = None
        self._max_radii2D = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def xyz_scales(self) -> torch.Tensor:
        return self._xyz_scales

    @property
    def xyz_scales_active(self) -> torch.Tensor:
        return torch.exp(self._xyz_scales)

    @property
    def rotation(self) -> torch.Tensor:
        return self._rotation

    @property
    def rotation_active(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation)

    @property
    def opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def opacity_active(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def features_dc(self) -> torch.Tensor:
        return self._features_dc

    def init(self, dataset: AbstractDataset):
        self._init_point_cloud_parameters(dataset.ply_data)

        if (
            self._enable_color_transform
            and self._color_transform_matrix_model_path_prior is not None
        ):
            self._init_color_transform_model()

        self._initialize_learning_rate()
        self._init_density_control()

    def load(self, pcd_path: str, dataset: AbstractDataset):
        plydata = PlyData.read(pcd_path)
        init_pcd_data = dataset.ply_data

        self._fetch_point_cloud_parameters(init_pcd_data, plydata)

        if self._enable_color_transform:
            self._fetch_color_transform_model(pcd_path)

        self._initialize_learning_rate()
        self._init_density_control()

    def save(self, pcd_path: str):
        self._save_point_cloud_parameters(pcd_path)

        if self._enable_color_transform:
            self._save_color_transform_parameters(
                pcd_path.replace(".ply", "_color_transform_matrix.pth")
            )

    def iteration_start(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        self._update_learning_rate(iteration)
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
            self._filter_invisible()

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        self._screenspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype, requires_grad=True, device=self.device
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
            antialiasing=False,
            debug=False,
        )
        rasterizer = GRzer(raster_settings=raster_settings)

        self._rendered_image, self._radii, self._rendered_depth = rasterizer(
            means3D=self.xyz,
            means2D=self._screenspace_points,
            shs=None,
            colors_precomp=self.features_dc,
            opacities=self.opacity_active,
            scales=self.xyz_scales_active,
            rotations=self.rotation_active,
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

    def get_regularization_loss(self, camera: Camera) -> torch.Tensor:
        loss = super().get_regularization_loss(camera)
        if self._enable_color_transform:
            loss += self._calculate_color_regularization_loss(camera)
        return loss

    def _init_point_cloud_parameters(self, pcd_data: BasicPointCloud):
        point_cloud = torch.tensor(np.asarray(pcd_data.points)).float().to(self._device)
        color = torch.tensor(np.asarray(pcd_data.colors)).float().to(self._device)

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
        self._xyz_scales = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opactities.requires_grad_(True))
        self._features_dc = nn.Parameter(color.contiguous().requires_grad_(True))

        self._init_outline_bounds(point_cloud)

    def _fetch_point_cloud_parameters(
        self, ply_data: PlyData, init_pcd_data: BasicPointCloud
    ):
        def SH2RGB(sh):
            C0 = 0.28209479177387814
            return sh * C0 + 0.5

        xyz_names = ["x", "y", "z"]
        xyz_scales_names = ["scale_" + str(i) for i in range(3)]
        rotation_names = ["rot_" + str(i) for i in range(4)]
        opacity_names = ["opacity"]
        features_dc_names = ["f_dc_" + str(i) for i in range(3)]

        xyz = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in xyz_names], axis=1
        )
        xyz_scales = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in xyz_scales_names],
            axis=1,
        )
        rotation = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in rotation_names], axis=1
        )
        opacity = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in opacity_names], axis=1
        )
        features_dc = np.stack(
            [np.asarray(ply_data.elements[0][name]) for name in features_dc_names],
            axis=1,
        )

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self._device).requires_grad_(
                True
            )
        )
        self._xyz_scales = nn.Parameter(
            torch.tensor(
                xyz_scales, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                rotation, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacity, dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(
                SH2RGB(features_dc), dtype=torch.float, device=self._device
            ).requires_grad_(True)
        )

        point_cloud = (
            torch.tensor(np.asarray(init_pcd_data.points)).float().to(self._device)
        )
        self._init_outline_bounds(point_cloud)

    def _save_point_cloud_parameters(self, pcd_path):
        def RGB2SH(rgb):
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().cpu().numpy()
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.xyz_scales.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        list_of_attributes = []
        list_of_attributes.extend(["x", "y", "z"])
        list_of_attributes.extend(["nx", "ny", "nz"])
        list_of_attributes.extend(["f_dc_" + str(i) for i in range(3)])
        list_of_attributes.extend(["opacity"])
        list_of_attributes.extend(["scale_" + str(i) for i in range(3)])
        list_of_attributes.extend(["rot_" + str(i) for i in range(4)])

        dtype_full = [(attribute, "f4") for attribute in list_of_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                RGB2SH(f_dc),
                opacity,
                scale,
                rotation,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        pcd_vertex_element = PlyElement.describe(elements, "vertex")
        PlyData([pcd_vertex_element]).write(pcd_path)

    def _init_color_transform_model(self):
        if os.path.exists(self._color_transform_matrix_model_path_prior):
            self._color_transformation_matrix_dict = torch.load(
                self._color_transform_matrix_model_path_prior
            )
            for matrix_param in self._color_transformation_matrix_dict.values():
                matrix_param.to(self._device)
        else:
            logging.warning(
                f"Color transformation matrix model not found at {self._color_transform_matrix_model_path_prior}"
            )

    def _fetch_color_transform_model(self, pcd_path: str):
        if self._color_transform_matrix_model_path_prior is None:
            self._color_transform_matrix_model_path_prior = pcd_path.replace(
                ".ply", "_color_transform_matrix.pth"
            )
        self._init_color_transform_model()

    def _apply_color_transformation(
        self, camera: Camera, image: torch.Tensor
    ) -> torch.Tensor:
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

        image_stack_with_alpha = torch.cat([image, torch.ones_like(image[:1])], dim=0)
        return torch.einsum("ij, jwh -> iwh", matrix_param, image_stack_with_alpha)

    def _apply_color_transformation_forward_only(
        self, camera: Camera, image: torch.Tensor
    ) -> torch.Tensor:
        camera_key = f"color_transformation_matrix_{camera.camera_info.camera_id}"
        if camera_key in self._color_transformation_matrix_dict:
            matrix_param = self._color_transformation_matrix_dict[camera_key]
            image_stack_with_alpha = torch.cat(
                [image, torch.ones_like(image[:1])], dim=0
            )
            return torch.einsum("ij, jwh -> iwh", matrix_param, image_stack_with_alpha)
        raise ValueError(
            f"Color transformation matrix not found for camera {camera_key}"
        )

    def _calculate_color_regularization_loss(self, camera: Camera) -> torch.Tensor:
        matrix_param = self._color_transformation_matrix_dict[
            f"color_transformation_matrix_{camera.camera_info.camera_id}"
        ]
        default_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float32,
            device=self._device,
        )
        return matrix_loss(
            matrix_param,
            default_matrix,
            torch.tensor(self._lambda_color_transform, device=self._device),
        )

    def _save_color_transform_parameters(self, path):
        torch.save(
            self._color_transformation_matrix_dict,
            path,
        )

    def _initialize_learning_rate(self):
        self._optimizer = torch.optim.Adam(self._initial_param_groups, lr=0.0, eps=1e-15)
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

    @property
    def _initial_param_groups(self) -> list:
        return [
            {
                "params": [self._xyz],
                "lr": self._learning_rate["xyz_init"] * self._cameras_extent,
                "name": "xyz",
            },
            {
                "params": [self._xyz_scales],
                "lr": self._learning_rate["xyz_scales"],
                "name": "xyz_scales",
            },
            {
                "params": [self._rotation],
                "lr": self._learning_rate["rotation"],
                "name": "rotation",
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

    def _update_learning_rate(self, iteration: int):
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self._xyz_scheduler_args(iteration)
            elif self._enable_color_transform and param_group["name"].startswith(
                "color_transformation_matrix"
            ):
                param_group["lr"] = self._color_transformation_scheduler_args(iteration)

    def _init_density_control(self):
        self._max_vspace_gradient = torch.zeros(
            (self._xyz.shape[0], 1), device=self._device
        )
        self._density_control_denom = torch.zeros(
            (self._xyz.shape[0], 1), device=self._device
        )
        self._max_radii2D = torch.zeros((self._xyz.shape[0]), device=self._device)

    def _cache_density_control(self):
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
                self._screenspace_points.grad[self._radii > 0, :2],
                dim=-1,
                keepdim=True,
            ),
        )
        self._density_control_denom[self._radii > 0] += 1

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
                new_xyz_scales = self._xyz_scales[space_mask].repeat(self._split_num, 1)
                new_rotation = self._rotation[space_mask].repeat(self._split_num, 1)
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
                    "xyz_scales": new_xyz_scales,
                    "rotation": new_rotation,
                    "opacity": new_opacity,
                    "features_dc": new_features_dc,
                }

                optimizable_tensors = self._extend_param_group(param_dict)

                self._xyz = optimizable_tensors["xyz"]
                self._xyz_scales = optimizable_tensors["xyz_scales"]
                self._rotation = optimizable_tensors["rotation"]
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
        self._opacity = self._reset_param_group({"opacity": new_opacity})["opacity"]

    def _init_outline_bounds(self, point_cloud):
        self._max_bounds_init = [torch.amax(point_cloud[:, i]) for i in range(3)]
        self._min_bounds_init = [torch.amin(point_cloud[:, i]) for i in range(3)]

    def _remove_outlier(self):
        max_bounds = torch.tensor(self._max_bounds_init, device=self._device)
        min_bounds = torch.tensor(self._min_bounds_init, device=self._device)
        mask_max = torch.any(self._xyz > max_bounds, dim=1)
        mask_min = torch.any(self._xyz < min_bounds, dim=1)
        mask = torch.logical_or(mask_max, mask_min)
        self._prune_points(mask)

    def _filter_invisible(self):
        self._prune_points(torch.sigmoid(self._opacity) < self._prune_threshold)

    def _prune_points(self, mask: torch.Tensor):
        mask = mask.squeeze()

        valid_point_mask = torch.logical_not(mask)

        param_list = [
            "xyz",
            "xyz_scales",
            "rotation",
            "opacity",
            "features_dc",
        ]
        param_dict = {param: valid_point_mask for param in param_list}

        gaussian_optimizable_tensors = self._prune_param_group(param_dict)
        self._xyz = gaussian_optimizable_tensors["xyz"]
        self._xyz_scales = gaussian_optimizable_tensors["xyz_scales"]
        self._rotation = gaussian_optimizable_tensors["rotation"]
        self._opacity = gaussian_optimizable_tensors["opacity"]
        self._features_dc = gaussian_optimizable_tensors["features_dc"]

        self._max_vspace_gradient = self._max_vspace_gradient[valid_point_mask]
        self._density_control_denom = self._density_control_denom[valid_point_mask]
        self._max_radii2D = self._max_radii2D[valid_point_mask]
        torch.cuda.empty_cache()

    def _add_param_group(self, param_dict: dict):
        for param_group in param_dict.values():
            self._optimizer.add_param_group(param_group)

    def _reset_param_group(self, param_dict: dict):
        gaussian_optimizable_tensors = {}
        for group in self._optimizer.param_groups:
            if group["name"] in param_dict:
                reset_value = param_dict[group["name"]]
                stored_state = self._optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(reset_value)
                stored_state["exp_avg_sq"] = torch.zeros_like(reset_value)
                del self._optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(reset_value.requires_grad_(True))
                self._optimizer.state[group["params"][0]] = stored_state
                gaussian_optimizable_tensors[group["name"]] = group["params"][0]
        return gaussian_optimizable_tensors

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

    def __len__(self) -> int:
        return self._xyz.shape[0] if self._xyz is not None else 0

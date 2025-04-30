import os
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Optional, Literal
import logging
import math
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from pydantic import BaseModel, field_validator, Field, ValidationInfo

from scene.dataset.abstract_dataset import AbstractDataset
from scene.model.abstract_gaussian_model import AbstractModel
from scene.cameras import Camera
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
)
from utils.graphics_utils import BasicPointCloud
from utils.optimizer_utils import (
    reset_param_group,
    extend_param_group,
    prune_param_group,
)
from utils.loss_utils import matrix_loss
from utils.sh_utils import RGB2SH

class BasicGaussianModelContext(BaseModel):
    device: str
    camera_extent: float = Field(..., gt=0)
    camera_count: int = Field(..., gt=0)
    camera_id_count: int = Field(..., gt=0)
    max_iterations: int = Field(..., gt=0)
    method_mask_loss: Literal["none", "cover", "penalty"]
    is_render_support_vspace: bool


class SchedulerLearningRateParams(BaseModel):
    init: float = Field(..., gt=0)
    final: float = Field(..., gt=0)
    delay_steps: Optional[int] = Field(None, ge=0)
    delay_mult: Optional[float] = Field(None, ge=0)
    max_steps: Optional[int] = Field(None, gt=0)

    @field_validator("delay_mult", mode="before")
    @classmethod
    def set_default_delay_mult(cls, value, info: ValidationInfo):
        if value is None:
            value = info.data.get("final") / info.data.get("init")
        return value

    @field_validator("max_steps", mode="before")
    @classmethod
    def set_default_max_steps(cls, value, info: ValidationInfo):
        if value is None:
            max_iterations = info.context.max_iterations
            return max_iterations
        return value

    @field_validator("delay_steps", mode="before")
    @classmethod
    def set_default_delay_steps(cls, value, info: ValidationInfo):
        if value is None:
            return max(100, info.context.camera_count)
        return value


class ColorTransformParams(BaseModel):
    enable: bool
    lambda_loss: float
    model_path_prior: Optional[str] = None
    lr: SchedulerLearningRateParams


class IterationParamsMixin(BaseModel):
    start: Optional[int] = Field(None, gt=0)
    step: Optional[int] = Field(None, gt=0)
    end: Optional[int] = Field(None, gt=0)

    @field_validator("start", mode="before")
    @classmethod
    def set_default_start(cls, value, info: ValidationInfo):
        if value is None:
            return max(2000, info.context.camera_count * 2)
        return value

    @field_validator("step", mode="before")
    @classmethod
    def set_default_step(cls, value, info: ValidationInfo):
        if value is None:
            return max(1000, info.context.camera_count)
        return value

    @field_validator("end", mode="before")
    @classmethod
    def set_default_end(cls, value, info: ValidationInfo):
        if value is None:
            return max(1, info.context.max_iterations // 2)
        return value


class DensityControlParams(IterationParamsMixin):
    enable: bool
    split_num: int = Field(..., gt=0)
    split_ratio: float = Field(..., gt=0, le=1)
    th_method: Literal["simple", "complex"] = "simple"
    grad_th_xyz: float = Field(..., gt=0)
    scale_th_xyz: float = Field(..., gt=0, le=1)

    @field_validator("start", mode="before")
    @classmethod
    def set_default_start(cls, value, info: ValidationInfo):
        if value is None:
            # Avoid density control conflicts with prune points
            return max(1999, info.context.camera_count * 2 - 1)
        return value

    @field_validator("step", mode="before")
    @classmethod
    def set_default_step(cls, value, info: ValidationInfo):
        if value is None:
            return max(60, info.context.camera_id_count * 2)
        return value


class ResetOpacityParams(IterationParamsMixin):
    enable: bool


class PrunePointsParams(IterationParamsMixin):
    enable: bool
    th_opacity: float
    th_radii: float

    @field_validator("step", mode="before")
    @classmethod
    def set_default_step(cls, value, info: ValidationInfo):
        if value is None:
            return max(30, info.context.camera_id_count, 30)
        return value


class LearningRateParams(BaseModel):
    xyz: SchedulerLearningRateParams
    xyz_scales: float
    rotation: float
    opacity: float
    features: float


class BasciGaussianModelParams(BaseModel):
    lambda_mask: float
    sh_degree: int = Field(..., ge=0, le=3)
    color_transform: ColorTransformParams
    density_control: DensityControlParams
    reset_opacity: ResetOpacityParams
    prune_points: PrunePointsParams
    learning_rate: LearningRateParams


class BasicGaussianModel(AbstractModel):
    def __init__(self, params: dict, context: dict):
        # Model parameters
        self._context = BasicGaussianModelContext.model_validate(context)
        self._basic_params = BasciGaussianModelParams.model_validate(
            params, context=self._context
        )

        # Constants
        self._device = torch.device(self._context.device)
        self._bg_color = torch.tensor(
            [0 for _ in range(9)],
            dtype=torch.float32,
            device=self.device,
        )

        # Optimization parameters
        self._xyz_scheduler_args = None
        self._color_transformation_scheduler_args = None
        self._optimizer = None

        # Other status
        self._active_sh_degree = 0

        # Last rendering results
        self._rendered_image = None
        self._rendered_depth = None

        # Last densification results to cache
        self._vspace_radii = None
        self._vspace_points = None
        self._vspace_values = None

        # Cached densification parameters
        self._max_vspace_grads = None
        self._max_vspace_radii = None
        self._densification_denom = None

        # Private optimizable parameters
        self.__color_transformation_matrix_dict = {}
        self.__xyz: torch.Tensor = None
        self.__xyz_scales: torch.Tensor = None
        self.__rotation: torch.Tensor = None
        self.__opacity: torch.Tensor = None
        self.__features_dc: torch.Tensor = None
        self.__features_rest: torch.Tensor = None

    def __len__(self) -> int:
        return self.xyz.shape[0] if self.xyz is not None else 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def sh_degree(self) -> int:
        return self._basic_params.sh_degree

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def xyz(self) -> torch.Tensor:
        return self.__xyz

    @property
    def xyz_scales(self) -> torch.Tensor:
        return self.__xyz_scales

    @property
    def xyz_scales_activated(self) -> torch.Tensor:
        return torch.exp(self.__xyz_scales)

    @property
    def rotation(self) -> torch.Tensor:
        return self.__rotation

    @property
    def rotation_activated(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.__rotation)

    @property
    def opacity(self) -> torch.Tensor:
        return self.__opacity

    @property
    def opacity_activated(self) -> torch.Tensor:
        return torch.sigmoid(self.__opacity)

    @property
    def features_dc(self) -> torch.Tensor:
        return self.__features_dc

    @property
    def features_rest(self) -> torch.Tensor:
        return self.__features_rest

    @property
    def features(self) -> torch.Tensor:
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    def init(self, dataset: AbstractDataset):
        self._init_point_cloud_parameters(dataset)

        if self._basic_params.color_transform.enable:
            self._init_color_transform_model()

        self._initialize_learning_rate()

    def load(self, pcd_path: str, dataset: AbstractDataset):
        self._fetch_point_cloud_parameters(PlyData.read(pcd_path))

        if self._basic_params.color_transform.enable:
            self._fetch_color_transform_model(pcd_path)

        self._initialize_learning_rate()

    def save(self, pcd_path: str):
        self._save_point_cloud_parameters(pcd_path)

        if self._basic_params.color_transform.enable:
            self._save_color_transform_parameters(
                pcd_path.replace(".ply", "_color_transform_matrix.pth")
            )

    def iteration_start(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        self._update_learning_rate(iteration)
        if self._basic_params.density_control.enable:
            self._density_control(iteration, camera, dataset)
        if iteration in range(
            self._basic_params.reset_opacity.start,
            self._basic_params.reset_opacity.end,
            self._basic_params.reset_opacity.step,
        ):
            self._reset_opacity(iteration, camera, dataset)
        if iteration in range(
            self._basic_params.prune_points.start,
            self._basic_params.prune_points.end,
            self._basic_params.prune_points.step,
        ):
            self._filter_points(iteration, camera, dataset)

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
        result = rasterizer(
            means3D=self.xyz,
            means2D=self._vspace_points,
            shs=self.features,
            colors_precomp=None,
            opacities=self.opacity_activated,
            scales=self.xyz_scales_activated,
            rotations=self.rotation_activated,
            cov3D_precomp=None,
        )

        assert len(result) == 3 + int(
            self._context.is_render_support_vspace
        ), "Invalid result length"
        if len(result) == 4:
            (
                self._rendered_image,
                self._vspace_radii,
                self._rendered_depth,
                self._vspace_values,
            ) = result
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
        return self.render(camera, GRsetting, GRzer)

    def get_regularization_loss(
        self, camera: Camera, dataset: AbstractDataset
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        if self._basic_params.color_transform.enable:
            loss += self._calculate_color_regularization_loss(camera)

        if self._context.method_mask_loss == "penalty":
            loss += self._calculate_mask_penalty_loss(camera)
        return loss

    def iteration_end(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        """Optional method to be called at the end of each iteration."""
        return

    def one_up_sh_degree(self):
        if self._active_sh_degree < self._basic_params.sh_degree:
            self._active_sh_degree += 1

    def _set_gaussian_attributes(self, param_dict: dict):
        assert all(
            key in param_dict
            for key in ["xyz", "xyz_scales", "rotation", "opacity", "features_dc", "features_rest"]
        ), "Missing required parameters in param_dict"
        assert all(
            param_dict["xyz"].shape[0] == param_dict[key].shape[0]
            for key in ["xyz_scales", "rotation", "opacity", "features_dc", "features_rest"]
        ), "Parameter shapes do not match"
        self.__xyz = param_dict["xyz"]
        self.__xyz_scales = param_dict["xyz_scales"]
        self.__rotation = param_dict["rotation"]
        self.__opacity = param_dict["opacity"]
        self.__features_dc = param_dict["features_dc"]
        self.__features_rest = param_dict["features_rest"]

    def _init_point_cloud_parameters(self, dataset: AbstractDataset):
        pcd_data = dataset.ply_data
        xyzs = torch.tensor(np.asarray(pcd_data.points)).float().to(self.device)
        color = torch.tensor(np.asarray(pcd_data.colors)).float().to(self.device)

        pcd_masks = torch.zeros(xyzs.shape[0], dtype=torch.bool, device=self.device)
        cameras = dataset.train_cameras + dataset.test_cameras
        for camera in cameras:
            pcd_masks = torch.logical_or(
                pcd_masks,
                camera.get_near_mask(
                    torch.tensor(dataset.ply_data.points, device=self.device)
                ),
            )
        prune_mask = ~pcd_masks
        xyzs = xyzs[prune_mask]
        color = color[prune_mask]

        dist2 = torch.clamp_min(distCUDA2(xyzs), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((xyzs.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opactities = inverse_sigmoid(
            0.1
            * torch.ones(
                (xyzs.shape[0], 1), dtype=torch.float, device=self.device
            )
        )

        features = torch.zeros(
            xyzs.shape[0], 3, (self._basic_params.sh_degree + 1) ** 2, device=self.device
        )
        features[:, :, 0] = RGB2SH(color)
        features_dc = features[:, :, 0:1].transpose(1, 2)
        features_rest = features[:, :, 1:].transpose(1, 2)

        self._set_gaussian_attributes(
            {
                "xyz": nn.Parameter(xyzs.contiguous().requires_grad_(True)),
                "xyz_scales": nn.Parameter(scales.requires_grad_(True)),
                "rotation": nn.Parameter(rots.requires_grad_(True)),
                "opacity": nn.Parameter(opactities.requires_grad_(True)),
                "features_dc": nn.Parameter(features_dc.requires_grad_(True)),
                "features_rest": nn.Parameter(features_rest.requires_grad_(True)),
            }
        )

        return prune_mask

    def _fetch_point_cloud_parameters(self, ply_data: PlyData):
        xyz_names = ["x", "y", "z"]
        xyz_scales_names = ["scale_" + str(i) for i in range(3)]
        rotation_names = ["rot_" + str(i) for i in range(4)]
        opacity_names = ["opacity"]

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

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(ply_data.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(ply_data.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(ply_data.elements[0]["f_dc_2"])
        extra_f_names = [
            p.name
            for p in ply_data.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        if len(extra_f_names) != 3 * (self._basic_params.sh_degree + 1) ** 2 - 3:
            raise ValueError(
                f"Invalid number of extra features. Expected {3 * (self._basic_params.sh_degree + 1) ** 2 - 3}, got {len(extra_f_names)}"
            )
        features_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
        for i, name in enumerate(extra_f_names):
            features_rest[:, i] = np.asarray(ply_data.elements[0][name])
        features_rest = features_rest.reshape(
            (features_rest.shape[0], 3, (self._basic_params.sh_degree + 1) ** 2 - 1)
        )

        self._active_sh_degree = self._basic_params.sh_degree
        self._set_gaussian_attributes(
            {
                "xyz": nn.Parameter(
                    torch.tensor(
                        xyz, dtype=torch.float, device=self.device
                    ).requires_grad_(True)
                ),
                "xyz_scales": nn.Parameter(
                    torch.tensor(
                        xyz_scales, dtype=torch.float, device=self.device
                    ).requires_grad_(True)
                ),
                "rotation": nn.Parameter(
                    torch.tensor(
                        rotation, dtype=torch.float, device=self.device
                    ).requires_grad_(True)
                ),
                "opacity": nn.Parameter(
                    torch.tensor(
                        opacity, dtype=torch.float, device=self.device
                    ).requires_grad_(True)
                ),
                "features_dc": nn.Parameter(
                    torch.tensor(
                        features_dc,
                        dtype=torch.float,
                        device=self.device,
                    )
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                ),
                "features_rest": nn.Parameter(
                    torch.tensor(
                        features_rest,
                        dtype=torch.float,
                        device=self.device,
                    )
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                ),
            }
        )

    def _save_point_cloud_parameters(self, pcd_path):
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
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

        list_of_attributes = []
        list_of_attributes.extend(["x", "y", "z"])
        list_of_attributes.extend(["nx", "ny", "nz"])
        list_of_attributes.extend(["f_dc_" + str(i) for i in range(3)])
        list_of_attributes.extend(
            ["f_rest_" + str(i) for i in range(3 * (self._basic_params.sh_degree + 1) ** 2 - 3)]
        )
        list_of_attributes.extend(["opacity"])
        list_of_attributes.extend(["scale_" + str(i) for i in range(3)])
        list_of_attributes.extend(["rot_" + str(i) for i in range(4)])

        dtype_full = [(attribute, "f4") for attribute in list_of_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
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
        if self._basic_params.color_transform.model_path_prior is None:
            return
        self._load_color_transform_model(
            self._basic_params.color_transform.model_path_prior
        )

    def _fetch_color_transform_model(self, pcd_path: str):
        if self._basic_params.color_transform.model_path_prior is not None:
            path = self._basic_params.color_transform.model_path_prior
        else:
            path = pcd_path.replace(".ply", "_color_transform_matrix.pth")
        self._load_color_transform_model(path)

    def _load_color_transform_model(self, path: str):
        if os.path.exists(path):
            logging.info(f"Loading color transformation matrix model from {path}")
            self.__color_transformation_matrix_dict = torch.load(path)
            for matrix_param in self.__color_transformation_matrix_dict.values():
                matrix_param.to(self.device)
        else:
            logging.warning(f"Color transformation matrix model not found at {path}")

    def _apply_color_transformation(
        self, camera: Camera, image: torch.Tensor
    ) -> torch.Tensor:
        camera_key = f"color_transformation_matrix_{camera.camera_info.camera_id}"
        if camera_key not in self.__color_transformation_matrix_dict:
            matrix_param = nn.Parameter(
                torch.tensor(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                requires_grad=True,
            )
            new_param_group = {
                "params": [matrix_param],
                "lr": self._basic_params.color_transform.lr.init,
                "name": camera_key,
            }
            self._optimizer.add_param_group(new_param_group)
            self.__color_transformation_matrix_dict[camera_key] = matrix_param
        else:
            matrix_param = self.__color_transformation_matrix_dict[camera_key]

        image_stack_with_alpha = torch.cat([image, torch.ones_like(image[:1])], dim=0)
        return torch.einsum("ij, jwh -> iwh", matrix_param, image_stack_with_alpha)

    def _apply_color_transformation_forward_only(
        self, camera: Camera, image: torch.Tensor
    ) -> torch.Tensor:
        camera_key = f"color_transformation_matrix_{camera.camera_info.camera_id}"
        if camera_key in self.__color_transformation_matrix_dict:
            matrix_param = self.__color_transformation_matrix_dict[camera_key]
            image_stack_with_alpha = torch.cat(
                [image, torch.ones_like(image[:1])], dim=0
            )
            return torch.einsum("ij, jwh -> iwh", matrix_param, image_stack_with_alpha)
        else:
            return image

    def _calculate_color_regularization_loss(self, camera: Camera) -> torch.Tensor:
        matrix_param = self.__color_transformation_matrix_dict[
            f"color_transformation_matrix_{camera.camera_info.camera_id}"
        ]
        default_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float32,
            device=self.device,
        )
        return matrix_loss(
            matrix_param,
            default_matrix,
            torch.tensor(
                self._basic_params.color_transform.lambda_loss, device=self.device
            ),
        )

    def _save_color_transform_parameters(self, path):
        torch.save(
            self.__color_transformation_matrix_dict,
            path,
        )

    @property
    def _initial_param_groups(self) -> list:
        return [
            {
                "params": [self.__xyz],
                "lr": self._basic_params.learning_rate.xyz.init
                * self._context.camera_extent,
                "name": "xyz",
            },
            {
                "params": [self.__xyz_scales],
                "lr": self._basic_params.learning_rate.xyz_scales,
                "name": "xyz_scales",
            },
            {
                "params": [self.__rotation],
                "lr": self._basic_params.learning_rate.rotation,
                "name": "rotation",
            },
            {
                "params": [self.__opacity],
                "lr": self._basic_params.learning_rate.opacity,
                "name": "opacity",
            },
            {
                "params": [self.__features_dc],
                "lr": self._basic_params.learning_rate.features,
                "name": "features_dc",
            },
            {
                "params": [self.__features_rest],
                "lr": self._basic_params.learning_rate.features / 20.0,
                "name": "features_rest",
            }
        ]

    def _initialize_learning_rate(self):
        self._optimizer = torch.optim.Adam(
            self._initial_param_groups, lr=0.0, eps=1e-15
        )
        self._xyz_scheduler_args = get_expon_lr_func(
            lr_init=self._basic_params.learning_rate.xyz.init
            * self._context.camera_extent,
            lr_final=self._basic_params.learning_rate.xyz.final
            * self._context.camera_extent,
            lr_delay_steps=self._basic_params.learning_rate.xyz.delay_steps,
            lr_delay_mult=self._basic_params.learning_rate.xyz.delay_mult,
            max_steps=self._basic_params.learning_rate.xyz.max_steps,
        )
        self._color_transformation_scheduler_args = get_expon_lr_func(
            lr_init=self._basic_params.color_transform.lr.init,
            lr_final=self._basic_params.color_transform.lr.final,
            lr_delay_steps=self._basic_params.color_transform.lr.delay_steps,
            lr_delay_mult=self._basic_params.color_transform.lr.delay_mult,
            max_steps=self._basic_params.color_transform.lr.max_steps,
        )

    def _update_learning_rate(self, iteration: int):
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self._xyz_scheduler_args(iteration)
            elif self._basic_params.color_transform.enable and param_group[
                "name"
            ].startswith("color_transformation_matrix"):
                param_group["lr"] = self._color_transformation_scheduler_args(iteration)

    def _init_density_control(self):
        self._max_vspace_grads = torch.zeros((len(self), 1), device=self.device)
        self._densification_denom = torch.zeros((len(self), 1), device=self.device)
        self._max_vspace_radii = torch.zeros((len(self)), device=self.device)

    def _cache_density_control(self):
        assert self._vspace_radii is not None and self._vspace_points is not None
        assert self._vspace_radii.shape[0] == len(self) and self._vspace_points.shape[
            0
        ] == len(self)
        self._max_vspace_radii[self._vspace_radii > 0] = torch.max(
            self._max_vspace_radii[self._vspace_radii > 0],
            self._vspace_radii[self._vspace_radii > 0],
        )
        self._max_vspace_grads[self._vspace_radii > 0] = torch.max(
            self._max_vspace_grads[self._vspace_radii > 0],
            torch.norm(
                self._vspace_points.grad[self._vspace_radii > 0, :2],
                dim=-1,
                keepdim=True,
            ),
        )
        self._densification_denom[self._vspace_radii > 0] += 1

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

            space_split_mask = torch.logical_and(
                space_mask,
                torch.max(self.xyz_scales_activated, dim=1).values
                > self._basic_params.density_control.scale_th_xyz
                * self._context.camera_extent,
            )
            space_split_mask_under_selection = torch.logical_and(
                space_split_mask, space_mask
            )[space_mask].repeat(split_num)

            new_xyz = self.xyz[space_mask].repeat(split_num, 1)
            new_xyz_scales = self.xyz_scales[space_mask].repeat(split_num, 1)
            new_rotation = self.rotation[space_mask].repeat(split_num, 1)
            new_opacity = self.opacity[space_mask].repeat(split_num, 1)
            new_features_dc = self.features_dc[space_mask].repeat(split_num, 1, 1)
            new_features_rest = self.features_rest[space_mask].repeat(split_num, 1, 1)

            stds = torch.exp(self.xyz_scales[space_split_mask]).repeat(split_num, 1)
            means = torch.zeros((stds.size(0), 3), device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self.rotation[space_split_mask]).repeat(
                split_num, 1, 1
            )
            new_xyz_random_offset = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_xyz[space_split_mask_under_selection] += new_xyz_random_offset

            new_xyz_scales[space_split_mask_under_selection] = torch.log(
                torch.exp(new_xyz_scales[space_split_mask_under_selection])
                * self._basic_params.density_control.split_ratio
            )

            new_mask = torch.zeros(
                split_num * space_mask.sum(),
                device=self.device,
                dtype=bool,
            )

            param_dict = {
                "xyz": new_xyz,
                "xyz_scales": new_xyz_scales,
                "rotation": new_rotation,
                "opacity": new_opacity,
                "features_dc": new_features_dc,
                "features_rest": new_features_rest,
            }

            optimizable_tensors = extend_param_group(self.optimizer, param_dict)
            self._set_gaussian_attributes(optimizable_tensors)
            self._max_vspace_radii = None
            self._max_vspace_grads = None
            self._densification_denom = None
            self._vspace_radii = None
            self._vspace_points = None
            self._prune_mask(torch.cat((space_mask, new_mask)))
            self._init_density_control()

    def _reset_opacity(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        new_opacity = inverse_sigmoid(
            0.1 * torch.ones((len(self), 1), dtype=torch.float, device=self.device)
        )
        opacity = reset_param_group(self.optimizer, {"opacity": new_opacity})["opacity"]
        self._set_gaussian_attributes(
            {
                "xyz": self.xyz,
                "xyz_scales": self.xyz_scales,
                "rotation": self.rotation,
                "opacity": opacity,
                "features_dc": self.features_dc,
                "features_rest": self.features_rest,
            }
        )

    def _filter_points(self, iteration: int, camera: Camera, dataset: AbstractDataset):
        mask_opacity = (
            self.opacity_activated < self._basic_params.prune_points.th_opacity
        ).squeeze()

        if self._basic_params.density_control.enable and iteration in range(
            self._basic_params.density_control.start
            + self._basic_params.density_control.step,
            self._basic_params.density_control.end,
            self._basic_params.density_control.step,
        ):
            logging.info(
                f"Skipping pruning large points due to density control step {iteration}. Please make sure your prune points step is not in the same range as density control step."
            )
            return
        else:
            mask_radii = self._vspace_radii > self._basic_params.prune_points.th_radii

        mask = torch.logical_or(mask_opacity, mask_radii)
        self._prune_mask(mask)

    def _prune_mask(self, mask: torch.Tensor):
        mask = mask.squeeze()
        valid_point_mask = torch.logical_not(mask)
        param_list = [
            "xyz",
            "xyz_scales",
            "rotation",
            "opacity",
            "features_dc",
            "features_rest",
        ]
        param_dict = {param: valid_point_mask for param in param_list}

        gaussian_optimizable_tensors = prune_param_group(self.optimizer, param_dict)
        self._set_gaussian_attributes(gaussian_optimizable_tensors)

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

        in_view_mask = torch.logical_and(in_bound_mask, non_zero_mask)
        if in_view_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        x_int = vspace_values[in_view_mask, 0].long()
        y_int = vspace_values[in_view_mask, 1].long()

        distances = camera.mask_distance_map[y_int, x_int].float()
        penalty[in_view_mask] = distances / torch.max(
            torch.tensor([W, H], device=self.device)
        )

        return self._basic_params.lambda_mask * torch.mean(penalty)

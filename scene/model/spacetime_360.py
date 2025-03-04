import torch
import numpy as np
from typing import Type
import math
import os
from plyfile import PlyData, PlyElement
from pydantic import BaseModel

from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from scene.cameras import Camera
from utils.general_utils import SH2RGB


class Spacetime360ModelConfig(BaseModel):
    ply_path: str


class Spacetime360Model(SpacetimeGaussianModel):

    def __init__(self, params: dict, context: dict):
        static_params = params["static"]
        non_static_params = {k: v for k, v in params.items() if k != "static"}
        super().__init__(non_static_params, context)
        self._static_params = Spacetime360ModelConfig.model_validate(
            static_params, context=context
        )

        self.__static_xyz: torch.Tensor = None
        self.__static_xyz_scales: torch.Tensor = None
        self.__static_rotation: torch.Tensor = None
        self.__static_opacity: torch.Tensor = None
        self.__static_features_dc: torch.Tensor = None
        self.__DEFAULT_STATIC_T: int = 0
        self.__DEFAULT_STATIC_T_SCALE: int = 1e6
        self.__DEFAULT_STATIC_MOTION: int = 0
        self.__DEFAULT_STATIC_OMEGA: int = 0

    @property
    def static_xyz(self):
        return self.__static_xyz

    @property
    def static_xyz_scales(self):
        return self.__static_xyz_scales

    @property
    def static_xyz_scales_activated(self):
        return torch.exp(self.__static_xyz_scales)

    @property
    def static_rotation(self):
        return self.__static_rotation

    @property
    def static_rotation_activated(self):
        return torch.nn.functional.normalize(self.__static_rotation)

    @property
    def static_opacity(self):
        return self.__static_opacity

    @property
    def static_opacity_activated(self):
        return torch.sigmoid(self.__static_opacity)

    @property
    def static_features_dc(self):
        return self.__static_features_dc

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        self._vspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype, requires_grad=True, device=self.device
        )
        self._vspace_points.retain_grad()
        self._static_vspce_points = torch.zeros_like(
            self.__static_xyz, dtype=self.__static_xyz.dtype, requires_grad=True, device=self.device
        )

        raster_settings = GRsetting(
            image_height=int(camera.resized_height),
            image_width=int(camera.resized_width),
            tanfovx=math.tan(camera.camera_info.FovX * 0.5),
            tanfovy=math.tan(camera.camera_info.FovY * 0.5),
            bg=self._bg_color,
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
                (self.xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=self.xyz.dtype,
                requires_grad=False,
                device=self.device,
            )
            - self.t
        )

        self._rendered_image, self._vspace_radii, self._rendered_depth = rasterizer(
            means3D=torch.cat((self.xyz_projected(delta_t), self.__static_xyz), dim=0),
            means2D=torch.cat((self._vspace_points, self._static_vspce_points), dim=0),
            shs=None,
            colors_precomp=torch.cat((self.features_dc, self.__static_features_dc), dim=0),
            opacities=torch.cat((self.opacity_activated_projected(delta_t), self.static_opacity_activated), dim=0),
            scales=torch.cat((self.xyz_scales_activated, self.static_xyz_scales_activated), dim=0),
            rotations=torch.cat((self.rotation_activated_projected(delta_t), self.static_rotation_activated), dim=0),
            cov3D_precomp=None,
        )

        if self._basic_params.color_transform.enable:
            self._rendered_image = self._apply_color_transformation(
                camera, self._rendered_image
            )
        
        self._vspace_radii = self._vspace_radii[: self.xyz.shape[0]]

        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def render_forward_only(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        screenspace_points = torch.zeros(
            (self.xyz.shape[0] + self.static_xyz.shape[0], self.xyz.shape[1]),
            dtype=self.xyz.dtype,
            requires_grad=True,
            device=self.device,
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
            sh_degree=0,
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
            trbfcenter=torch.cat((self.t, torch.full((self.static_xyz.shape[0], self.t.shape[1]), self.__DEFAULT_STATIC_T, device=self.device)), dim=0),
            trbfscale=torch.cat((self.t_scale, torch.full((self.static_xyz.shape[0], self.t_scale.shape[1]), self.__DEFAULT_STATIC_T_SCALE, device=self.device)), dim=0),
            motion=torch.cat((self.motion_full_degree, torch.full((self.static_xyz.shape[0], self.motion_full_degree.shape[1]), self.__DEFAULT_STATIC_MOTION, device=self.device)), dim=0),
            means3D=torch.cat((self.xyz, self.static_xyz), dim=0),
            means2D=screenspace_points,
            shs=None,
            colors_precomp=torch.cat((self.features_dc, self.static_features_dc), dim=0),
            opacities=torch.cat((self.opacity_activated_projected(delta_t), self.static_opacity_activated), dim=0),
            scales=torch.cat((self.xyz_scales_activated, self.static_xyz_scales_activated), dim=0),
            rotations=torch.cat((self.rotation_activated_projected(delta_t), self.static_rotation_activated), dim=0),
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

    def _set_static_gaussian_attributes(self, param_dict: dict):
        assert all(
            key in param_dict for key in ["xyz", "xyz_scales", "rotation", "opacity", "features_dc"]
        )
        assert all(
            param_dict["xyz"].shape[0] == param_dict[key].shape[0]
            for key in ["xyz_scales", "rotation", "opacity", "features_dc"]
        )
        self.__static_xyz = param_dict["xyz"]
        self.__static_xyz_scales = param_dict["xyz_scales"]
        self.__static_rotation = param_dict["rotation"]
        self.__static_opacity = param_dict["opacity"]
        self.__static_features_dc = param_dict["features_dc"]

    def _init_point_cloud_parameters(self, pcd_data):
        self._load_static_point_cloud()
        return super()._init_point_cloud_parameters(pcd_data)

    def _fetch_point_cloud_parameters(self, ply_data, is_sh=False):
        self._load_static_point_cloud()
        return super()._fetch_point_cloud_parameters(ply_data, is_sh)

    def _load_static_point_cloud(self, is_sh: bool = True):

        static_ply_path = self._static_params.ply_path
        ply_data = PlyData.read(static_ply_path)

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

        param_dict = {
            "xyz": torch.tensor(xyz, dtype=torch.float, device=self.device),
            "xyz_scales": torch.tensor(
                xyz_scales, dtype=torch.float, device=self.device
            ),
            "rotation": torch.tensor(
                rotation, dtype=torch.float, device=self.device
            ),
            "opacity": torch.tensor(
                opacity, dtype=torch.float, device=self.device
            ),
            "features_dc": torch.tensor(
                SH2RGB(features_dc) if is_sh else features_dc,
                dtype=torch.float,
                device=self.device,
            ),
        }
        self._set_static_gaussian_attributes(param_dict)

    def _save_point_cloud_parameters(self, pcd_path):
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
        xyz = torch.cat(
            [self.xyz.cpu().detach(), self.static_xyz.cpu().detach()]
        ).numpy()
        trbf_center = torch.cat(
            [self.t.cpu().detach(), torch.full((self.static_xyz.shape[0], self.t.shape[1]), self.__DEFAULT_STATIC_T).cpu().detach()]
        ).numpy()
        trbf_scale = torch.cat(
            [self.t_scale.cpu().detach(), torch.full((self.static_xyz.shape[0], self.t_scale.shape[1]), self.__DEFAULT_STATIC_T_SCALE).cpu().detach()]
        ).numpy()
        normals = np.zeros_like(xyz)
        motion_full_degree = torch.cat(
            [self.motion_full_degree.cpu().detach(), torch.full((self.static_xyz.shape[0], self.motion_full_degree.shape[1]), self.__DEFAULT_STATIC_MOTION).cpu().detach()]
        ).numpy()
        f_dc = torch.cat(
            [self.features_dc.cpu().detach(), self.static_features_dc.cpu().detach()]
        ).numpy()
        opacity = torch.cat(
            [self.opacity.cpu().detach(), self.static_opacity.cpu().detach()]
        ).numpy()
        scale = torch.cat(
            [self.xyz_scales.cpu().detach(), self.static_xyz_scales.cpu().detach()]
        ).numpy()
        rotation = torch.cat(
            [self.rotation.cpu().detach(), self.static_rotation.cpu().detach()]
        ).numpy()
        omega_full_degree = torch.cat(
            [self.omega_full_degree.cpu().detach(), torch.full((self.static_xyz.shape[0], self.omega_full_degree.shape[1]), self.__DEFAULT_STATIC_OMEGA).cpu().detach()]
        ).numpy()

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
                motion_full_degree,
                f_dc,
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

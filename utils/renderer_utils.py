from importlib import import_module
from typing import Literal
from scene.cameras import Camera
from typing import Type
import torch
import math
from scene.model.st_gaussian_model import STGWithSTModel
from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from scene.model.basic_gaussian_model import BasicGaussianModel
from utils.math_utils import trbfunction
from utils.general_utils import inverse_sigmoid

RendererNames = Literal[
    "diff_gaussian_rasterization",
    "forward_lite",
]

__RENDERER_ATTRIBUTES = {
    "diff_gaussian_rasterization": {"forward_only": False, "support_vspace": False},
    "forward_lite": {"forward_only": True, "support_vspace": False},
}

def get_renderer_attributes(renderer: RendererNames) -> dict:
    return __RENDERER_ATTRIBUTES[renderer]


def is_forward_only(renderer: RendererNames) -> bool:
    return get_renderer_attributes(renderer)["forward_only"]

def is_support_vspace(renderer: RendererNames) -> bool:
    return get_renderer_attributes(renderer)["support_vspace"]

def parse_renderer(renderer: RendererNames) -> tuple:
    rasterizer_module = import_module(renderer)
    return getattr(rasterizer_module, "GaussianRasterizer"), getattr(
        rasterizer_module, "GaussianRasterizationSettings"
    )

def xyz_projected(xyz : torch.Tensor, motion : torch.Tensor, motion_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    output = xyz.clone()
    for degree in range(1, motion_degree + 1):
        output = (
            output + motion[:, (degree - 1) * 3 : degree * 3] * delta_t**degree
        )
    return output

def rotation_projected(rotation : torch.Tensor, omega : torch.Tensor, omega_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    output = rotation.clone()
    for degree in range(1, omega_degree + 1):
        output = (
            output + omega[:, (degree - 1) * 4 : degree * 4] * delta_t**degree
        )
    return output

def rotation_activated_projected(rotation : torch.Tensor, omega : torch.Tensor, omega_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(rotation_projected(rotation, omega, omega_degree, delta_t))

def opacity_projected(opacity_activated : torch.Tensor, t_scale_active : torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    return inverse_sigmoid(opacity_activated_projected(opacity_activated, t_scale_active, delta_t))

def opacity_activated_projected(opacity_activated : torch.Tensor, t_scale_active : torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    return opacity_activated * trbfunction(delta_t / t_scale_active)

def render(gaussian, *args, **kwargs) -> dict:
    if isinstance(gaussian, SpacetimeGaussianModel) or isinstance(gaussian, STGWithSTModel):
        return render_stg(gaussian, *args, **kwargs)
    elif isinstance(gaussian, BasicGaussianModel):
        return render_basic(gaussian, *args, **kwargs)
    else:
        print("Render not supported")

# TODO: Combine render_basic and render_stg
def render_basic(gaussian : BasicGaussianModel, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
    gaussian._vspace_points = torch.zeros_like(
        gaussian.xyz, dtype=gaussian.xyz.dtype, requires_grad=True, device=gaussian.device
    )
    gaussian._vspace_points.retain_grad()
    raster_settings = GRsetting(
        image_height=int(camera.resized_height),
        image_width=int(camera.resized_width),
        tanfovx=math.tan(camera.camera_info.FovX * 0.5),
        tanfovy=math.tan(camera.camera_info.FovY * 0.5),
        bg=gaussian._bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussian._active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        antialiasing=False,
        debug=False,
    )
    rasterizer = GRzer(raster_settings=raster_settings)
    result = rasterizer(
        means3D=gaussian.xyz,
        means2D=gaussian._vspace_points,
        shs=gaussian.features,
        colors_precomp=None,
        opacities=gaussian.opacity_activated,
        scales=gaussian.xyz_scales_activated,
        rotations=gaussian.rotation_activated,
        cov3D_precomp=None,
    )

    assert len(result) == 3 + int(
        gaussian._context.is_render_support_vspace
    ), "Invalid result length"
    if len(result) == 4:
        (
            rendered_image,
            gaussian._vspace_radii,
            rendered_depth,
            gaussian._vspace_values,
        ) = result
    elif len(result) == 3:
        rendered_image, gaussian._vspace_radii, rendered_depth = result
    else:
        raise ValueError("Invalid result length")

    if gaussian._basic_params.color_transform.enable:
        rendered_image = gaussian._apply_color_transformation(
            camera, rendered_image
        )

    return {
        "rendered_image": rendered_image,
        "depth": rendered_depth,
    }

def from_camera(camera: Camera, bg_color : torch.Tensor, active_sh_degree : int, GRsetting : Type):
    return GRsetting(
        image_height=int(camera.resized_height),
        image_width=int(camera.resized_width),
        tanfovx=math.tan(camera.camera_info.FovX * 0.5),
        tanfovy=math.tan(camera.camera_info.FovY * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        antialiasing=False,
        debug=False,
    )

def render_stg(gaussian : SpacetimeGaussianModel, camera : Camera, GRsetting: Type, GRzer: Type) -> dict:
    timestamp_ratio = camera.camera_info.timestamp_ratio 
    gaussian.update(timestamp_ratio)
    print(f"At render time {timestamp_ratio}, rendering {gaussian.xyz.shape[0]} gaussians")
    
    gaussian._vspace_points = torch.zeros_like(
        gaussian.xyz, dtype=gaussian.xyz.dtype, requires_grad=True, device=gaussian.device
    )

    gaussian._vspace_points.retain_grad()

    raster_settings = from_camera(camera, gaussian._bg_color, gaussian._active_sh_degree, GRsetting)
    
    rasterizer = GRzer(raster_settings=raster_settings)

    delta_t = (
        torch.full(
            (gaussian.xyz.shape[0], 1),
            camera.camera_info.timestamp_ratio,
            dtype=gaussian.xyz.dtype,
            requires_grad=False,
            device=gaussian.device,
        )
        - gaussian.t
    )

    means3D = xyz_projected(gaussian.xyz, gaussian.motion, gaussian._spacetime_params.motion_degree, delta_t)
    opacities = opacity_activated_projected(gaussian.opacity_activated, gaussian.t_scale_active, delta_t)
    rotations = rotation_activated_projected(gaussian.rotation, gaussian.omega, gaussian._spacetime_params.omega_degree, delta_t)

    result = rasterizer(
        means3D=means3D,
        means2D=gaussian._vspace_points,
        shs=gaussian.features,
        colors_precomp=None,
        opacities=opacities,
        scales=gaussian.xyz_scales_activated,
        rotations=rotations,
        cov3D_precomp=None,
    )

    assert len(result) == 3 + int(gaussian._context.is_render_support_vspace), "Invalid result length"
    if len(result) == 4:
        rendered_image, gaussian._vspace_radii, rendered_depth, gaussian._vspace_values = result
    elif len(result) == 3:
        rendered_image, gaussian._vspace_radii, rendered_depth = result
    else:
        raise ValueError("Invalid result length")

    if gaussian._basic_params.color_transform.enable:
        rendered_image = gaussian._apply_color_transformation(
            camera, rendered_image
        )

    return {
        "rendered_image": rendered_image,
        "depth": rendered_depth,
    }
from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from utils.st_utils import SegmentTree, Interval
import torch
from scene.dataset.abstract_dataset import AbstractDataset
from scene.cameras import Camera
from typing import Type
import torch.nn as nn
import math
from utils.math_utils import trbfunction
import time

class GaussianSegment:
    def __init__(self, interval : Interval):
        self.interval = interval
        self.gaussian_ids : None | torch.Tensor = None

    def fill(self, gaussian_ids : torch.Tensor):
        self.gaussian_ids = gaussian_ids

    def size(self):
        return self.gaussian_ids.shape[0]
    
def xyz_projected(xyz: torch.Tensor, motion: torch.Tensor, motion_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    output = xyz.clone()
    for degree in range(1, motion_degree + 1):
        output = (
            output + motion[:, (degree - 1) * 3 : degree * 3] * delta_t**degree
        )
    return output
    
def opacity_activated_projected(opacity_activated : torch.Tensor, t_scale_activated : torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    return opacity_activated * trbfunction(delta_t / t_scale_activated)

def rotation_projected(rotation : torch.Tensor , omega : torch.Tensor, omega_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    output = rotation.clone()
    for degree in range(1, omega_degree + 1):
        output = (
            output + omega[:, (degree - 1) * 4 : degree * 4] * delta_t**degree
        )
    return output

def rotation_activated_projected(rotation : torch.Tensor, omega : torch.Tensor, omega_degree : int, delta_t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(rotation_projected(rotation, omega, omega_degree, delta_t))

class STreeModel(SpacetimeGaussianModel):
    def __init__(self, params: dict, context: dict):
        self.max_level = 3
        self.max_duration = 1.0
        self._st = SegmentTree(self.max_level, self.max_duration, GaussianSegment)
        super().__init__(params, context)

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        # render_start = time.time()
        use_mask = 1
        if use_mask:
            # start = time.time()
            mask = self._get_active_gaussians_mask_at_t(camera._camera_info.timestamp_ratio)
            # print("Time took to get mask ", time.time() - start)
            # start_mask = time.time()
            xyz = self.xyz[mask]
            t_center = self.t[mask]
            features = self.features[mask]
            scales = self.xyz_scales_activated[mask]
            motion = self.motion[mask]
            t_scale_activated = self.t_scale_active[mask]
            opacity_activated = self.opacity_activated[mask]
            rotation = self.rotation[mask]
            omega = self.omega[mask]
            # print("Masking took ", time.time() - start_mask)
            # Mask takes 0.02 seconds
            # This is 0.03 seconds per render
        else:
            # This is 0.002 seconds per render
            xyz = self.xyz
            t_center = self.t
            features = self.features
            scales = self.xyz_scales_activated
            motion = self.motion
            t_scale_activated = self.t_scale_active
            opacity_activated = self.opacity_activated
            rotation = self.rotation
            omega = self.omega

        self._vspace_points = torch.zeros_like(
            xyz, dtype=xyz.dtype, requires_grad=True, device=self.device
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
                (xyz.shape[0], 1),
                camera.camera_info.timestamp_ratio,
                dtype=xyz.dtype,
                requires_grad=False,
                device=self.device,
            )
            - t_center
        )

        means3D = xyz_projected(xyz, motion, self._spacetime_params.motion_degree, delta_t)
        opacities = opacity_activated_projected(opacity_activated, t_scale_activated, delta_t)
        rotations = rotation_activated_projected(rotation, omega, self._spacetime_params.omega_degree, delta_t)

        result = rasterizer(
            means3D=means3D,
            means2D=self._vspace_points,
            shs=features,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
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
        # print("Render took ", time.time() - render_start)
        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def load(self, pcd_path: str, dataset: AbstractDataset):
        super().load(pcd_path, dataset)
        self._rearrange_to_segments()

    def _rearrange_to_segments(self):
        ids_arranged_per_segment = torch.cat(get_ids_arranged_per_segment(self._st, self.opacity_activated, self.t, self.t_scale_active)).to(self.device)

        attr_list = [
            "xyz",
            "xyz_scales",
            "rotation",
            "opacity",
            "features_dc",
            "features_rest",
            "t",
            "t_scale",
            "motion",
            "omega",
        ]

        param_dict = {}
        # Rearrange all gaussian attributes such that they lie next to each other
        # e.g. [Gaussian_segment1, Gsegment2, Gsegment3, ...]
        for attr in attr_list:
            param_dict[attr] = nn.Parameter(getattr(self, attr)[ids_arranged_per_segment])
        self._set_gaussian_attributes_all(param_dict)

    def _get_active_gaussians_mask_at_t(self, t : float):
        """
        Returns a mask of active gaussians at time t
        """
        gaussian_mask = torch.zeros((self.xyz.shape[0], ), dtype=torch.bool)
        active_gaussians_id = torch.cat(self._st.get_active_gaussians_id_at_time(t))
        gaussian_mask[active_gaussians_id] = True
        return gaussian_mask
    

def get_gaussian_interval(opacity_activated : torch.Tensor, tcenter : torch.Tensor, tscale_activated : torch.Tensor, eps = 1e-5) -> torch.Tensor:
    # Let the interval be [t_left, t_right].
    # According to the Gaussian function, the opacity at time $t$ is
    # $$O(t) = opcaity * \exp(-\frac{(t - tcenter)^2}{2 * tscale^2})$$
    # In most cases, the Gaussian is considered to be invisible if
    # $O(t) < 0.005$. Therefore, we can find the left and right bounds
    # $$t_left = tcenter - \sqrt{-2 * tscale^2 * \log(0.005 / Opacity)}$$
    # $$t_right = tcenter + \sqrt{-2 * tscale^2 * \log(0.005 / Opacity)}$$
    # The interval is [t_left, t_right].
    half = torch.where(opacity_activated > 0.005, tscale_activated * torch.sqrt(-2.0 * torch.log(0.005 / opacity_activated)), eps)
    return torch.cat((tcenter - half, tcenter + half), dim=-1)

def get_ids_arranged_per_segment(segment_tree : SegmentTree, opacity_activated : torch.Tensor, t_center : torch.Tensor, t_scale_activated : torch.Tensor) -> None:
    """
    Method returns tensor of same shape as input attributes, where it contains  flattened
    [Gaussian_segment1_ids, Gsegment2_ids, Gsegment3_ids, ...]
    """
    num_gaussians = opacity_activated.shape[0]
    gaussian_intervals = get_gaussian_interval(opacity_activated, t_center, t_scale_activated)
    gaussian_segment_ids = torch.zeros(num_gaussians)
    level_segment_start_id = 1
    for level in range(1, segment_tree.max_level+1):
        ranges = torch.linspace(0.0, segment_tree.duration, (2 ** level)+1)
        bounds = torch.stack((ranges[:-1], ranges[1:]), dim=-1)
        num_segment_bounds = len(bounds)
        for segment_id in range(num_segment_bounds):
            bound = bounds[segment_id]
            is_in_bounds = (bound[0]  <= gaussian_intervals[:, 0]) & (bound[1] > gaussian_intervals[:, 1])
            nonzero_indices = torch.nonzero(is_in_bounds)
            segment_linear_id = level_segment_start_id + segment_id
            gaussian_segment_ids[nonzero_indices] = segment_linear_id
            
        level_segment_start_id += num_segment_bounds

    print("Total number of gaussians ", num_gaussians)
    ids_arranged_per_segment = []
    num_gaussians_acc = 0
    for i, segment in enumerate(segment_tree.get_all_segments_ref()):
        segment_ids = (gaussian_segment_ids == i).nonzero().squeeze(-1)
        segment.fill(torch.arange(num_gaussians_acc, num_gaussians_acc + segment_ids.shape[0], dtype=torch.int32))
        num_gaussians_acc += segment_ids.shape[0]
        ids_arranged_per_segment.append(segment_ids)
        print(f"Segment {i} size is {segment_ids.shape[0]}")

    return ids_arranged_per_segment
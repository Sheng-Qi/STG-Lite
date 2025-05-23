from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from utils.st_utils import SegmentTree, Interval
import torch
from scene.dataset.abstract_dataset import AbstractDataset
from scene.cameras import Camera
from typing import Type
import math

class GaussianSegment:
    def __init__(self, interval : Interval):
        self.interval = interval
        self.gaussian_ids : None | torch.Tensor = None

    def fill(self, gaussian_ids : torch.Tensor):
        self.gaussian_ids = gaussian_ids

    def size(self):
        return self.gaussian_ids.shape[0]
    
class STreeModel(SpacetimeGaussianModel):
    def __init__(self, params: dict, context: dict):
        self.max_level = 3
        self.max_duration = 1.0
        self._st = SegmentTree(self.max_level, self.max_duration, GaussianSegment)
        super().__init__(params, context)

    def render(self, camera: Camera, GRsetting: Type, GRzer: Type) -> dict:
        mask = self._get_active_gaussians_mask_at_t(camera._camera_info.timestamp_ratio)
        xyz = self.xyz[mask]
        t_center = self.t[mask]

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

        means3D = self.xyz_projected(delta_t)[mask]
        features = self.features[mask]
        opacities = self.opacity_activated_projected(delta_t)[mask]
        scales = self.xyz_scales_activated[mask]
        rotations = self.rotation_activated_projected(delta_t)[mask]
        
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

        return {
            "rendered_image": self._rendered_image,
            "depth": self._rendered_depth,
        }

    def load(self, pcd_path: str, dataset: AbstractDataset):
        super().load(pcd_path, dataset)
        self._divide_to_segments()

    def _divide_to_segments(self):
        segment_gaussians_to_st(self._st, self.opacity_activated, self.t, self.t_scale_active)

    def _get_active_gaussians_mask_at_t(self, t : float):
        """
        Returns a mask of active gaussians at time t
        """
        gaussian_mask = torch.zeros((self.xyz.shape[0], ), dtype=torch.int32)
        active_gaussians_id = torch.cat(self._st.get_active_gaussians_id_at_time(t))
        gaussian_mask[active_gaussians_id] = 1
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

def segment_gaussians_to_st(segment_tree : SegmentTree, opacity_activated : torch.Tensor, t_center : torch.Tensor, t_scale_activated : torch.Tensor) -> None:
    """
    Method returns None, fills in segment tree's segments. Each segment contains a gaussian id 
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
    for i, segment in enumerate(segment_tree.get_all_segments_ref()):
        segment_ids = (gaussian_segment_ids == i).nonzero().squeeze(-1)
        segment.fill(segment_ids)
        print(f"Segment {i} size is {segment.size()}")
from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from utils.st_utils import SegmentTree, Interval
from typing import NamedTuple
import torch
import time

def get_gaussian_interval(opacity_activated : torch.Tensor, tcenter : torch.Tensor, tscale_activated : torch.Tensor, eps = 1e-5) -> Interval:
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


class GaussianAttrs(NamedTuple):
    xyz : torch.Tensor
    xyz_scales : torch.Tensor
    rotation : torch.Tensor
    opacity : torch.Tensor
    features_dc : torch.Tensor
    features_rest : torch.Tensor
    t : torch.Tensor
    t_scale : torch.Tensor
    motion : torch.Tensor
    omega : torch.Tensor

class GaussianSegment:
    def __init__(self, interval : Interval):
        self.interval = interval

        self.xyz : None | torch.Tensor = None
        self.xyz_scales : None | torch.Tensor = None
        self.rotation : None | torch.Tensor = None
        self.opacity : None | torch.Tensor = None
        self.features_dc : None | torch.Tensor = None
        self.features_rest : None | torch.Tensor = None

        self.t : None | torch.Tensor = None
        self.t_scale : None | torch.Tensor = None
        self.motion : None | torch.Tensor = None
        self.omega : None | torch.Tensor = None

    def fill(self, xyz, xyz_scales, rotation, opacity, features_dc, features_rest, t, t_scale, motion, omega):
        self.xyz = xyz
        self.xyz_scales = xyz_scales
        self.rotation = rotation
        self.opacity = opacity
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.t = t
        self.t_scale = t_scale
        self.motion = motion
        self.omega = omega

    def size(self):
        return len(self.xyz)

class STGWithSTModel(SpacetimeGaussianModel):
    def __init__(self, params: dict, context: dict):
        self.max_level = 1
        self.max_duration = 1.0
        self._st = SegmentTree(self.max_level, self.max_duration, GaussianSegment)
        super().__init__(params, context)

        self.attr_list = [
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

    def deallocate(self):
        """
        Offload all Gaussian attributes from GPU        
        and divide to segments
        """
        self.xyz.detach().cpu()
        self.xyz_scales.detach().cpu()
        self.rotation.detach().cpu()
        self.opacity.detach().cpu()
        self.features_dc.detach().cpu()
        self.features_rest.detach().cpu()

        self.t.detach().cpu()
        self.t_scale.detach().cpu()
        self.motion.detach().cpu()
        self.omega.detach().cpu()

        print("Deallocated")

        self._divide_to_segments()

    def _divide_to_segments(self):
        """
        Move all gaussians to their respective segments, stored in self._st        
        """
        gaussian_segment_ids = self._get_segment_ids(self.opacity_activated, self.t, self.t_scale_active)
        for i, segment in enumerate(self._st.get_all_segments_ref()):
            segment_ids = (gaussian_segment_ids == i).nonzero().squeeze(-1)
            segment.fill(*(getattr(self, attr)[segment_ids] for attr in self.attr_list))
            print(f"Segment {i} size is {segment_ids.shape}")

    def _get_segment_ids(self, opacity_activated : torch.Tensor, t_center : torch.Tensor, t_scale_activated : torch.Tensor):
        gaussian_intervals = get_gaussian_interval(opacity_activated, t_center, t_scale_activated)
        num_gaussians = opacity_activated.shape[0]
        gaussian_segment_ids = torch.zeros(num_gaussians)
        start_idx = 1
        for level in range(1, self.max_level+1):
            ranges = torch.linspace(0.0, self.max_duration, (2 ** level)+1)
            bounds = torch.stack((ranges[:-1], ranges[1:]), dim=-1)
            for i in range(len(bounds)):
                bound = bounds[i]
                is_in_bounds = (bound[0]  <= gaussian_intervals[:, 0]) & (bound[1] > gaussian_intervals[:, 1])
                nonzero_indices = torch.nonzero(is_in_bounds)
                gaussian_segment_ids[nonzero_indices] = start_idx + i
            start_idx *= 2
        return gaussian_segment_ids

    def update(self, time : float):
        """
        Gather all relevant segments and combine
        """
        gaussian_segments : list[GaussianSegment] = self._st.get_at_time(time)
        
        gather_from_segments : torch.Tensor = lambda attr : torch.cat(tuple(getattr(seg, attr) for seg in gaussian_segments if seg.size() > 0)).to(self.device)        

        self._set_gaussian_attributes_all({attr : gather_from_segments(attr).to(self.device) for attr in self.attr_list})


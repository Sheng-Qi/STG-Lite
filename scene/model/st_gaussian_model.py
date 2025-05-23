from scene.model.spacetime_gaussian_model import SpacetimeGaussianModel
from utils.st_utils import SegmentTree, Interval
from typing import NamedTuple
import torch

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
        self.gaussian_ids : None | torch.Tensor = None

    def fill(self, gaussian_ids : torch.Tensor):
        self.gaussian_ids = gaussian_ids

    def size(self):
        return self.gaussian_ids.shape[0]

class STGWithSTModel(SpacetimeGaussianModel):
    def __init__(self, params: dict, context: dict):
        self.max_level = 3
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


    def divide_to_segments(self):
        segment_gaussians_to_st(self._st, self.opacity_activated, self.t, self.t_scale_active)

    def get_active_gaussians_mask_at_t(self, t : float):
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
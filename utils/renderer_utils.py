from importlib import import_module
from typing import Literal

RendererNames = Literal[
    "diff_gaussian_rasterization",
    "diff_gaussian_rasterization_custom",
    "forward_lite",
]

__RENDERER_ATTRIBUTES = {
    "diff_gaussian_rasterization": {"forward_only": False, "support_vspace": False},
    "diff_gaussian_rasterization_custom": {
        "forward_only": False,
        "support_vspace": True,
    },
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

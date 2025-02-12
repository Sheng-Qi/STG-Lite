import torch

def trbfunction(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-1 * x.pow(2))

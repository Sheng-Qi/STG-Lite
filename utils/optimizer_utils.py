import torch
import torch.nn as nn

def reset_param_group(optimizer: torch.optim.Optimizer, param_dict: dict) -> dict:
    gaussian_optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] in param_dict:
            reset_value = param_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            stored_state["exp_avg"] = torch.zeros_like(reset_value)
            stored_state["exp_avg_sq"] = torch.zeros_like(reset_value)
            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(reset_value.requires_grad_(True))
            optimizer.state[group["params"][0]] = stored_state
            gaussian_optimizable_tensors[group["name"]] = group["params"][0]
    return gaussian_optimizable_tensors

def prune_param_group(optimizer: torch.optim.Optimizer, param_dict: dict) -> dict:
    gaussian_optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] in param_dict:
            mask = param_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                optimizer.state[group["params"][0]] = stored_state
                gaussian_optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                gaussian_optimizable_tensors[group["name"]] = group["params"][0]
    return gaussian_optimizable_tensors

def extend_param_group(optimizer: torch.optim.Optimizer, param_dict: dict) -> dict:
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] in param_dict:
            extension_tensor = param_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
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

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

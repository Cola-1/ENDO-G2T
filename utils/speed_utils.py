import os
import json
import math
import torch
import numpy as np
from typing import List, Tuple, Dict


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_active_indices(path: str, active_indices: torch.Tensor):
    ensure_dir(os.path.dirname(path))
    np.save(path, active_indices.detach().cpu().numpy())


def load_active_indices(path: str) -> torch.Tensor:
    arr = np.load(path)
    return torch.from_numpy(arr).long()


def choose_keyframe_times(time_min: float, time_max: float, delta_t: float) -> List[float]:
    if delta_t <= 0:
        return [time_min, time_max]
    num = max(2, int(math.ceil((time_max - time_min) / delta_t)) + 1)
    return list(np.linspace(time_min, time_max, num=num, dtype=np.float32))


def merge_active_masks(indices_list: List[torch.Tensor], num_pts: int) -> torch.Tensor:
    if len(indices_list) == 0:
        return torch.zeros((0,), dtype=torch.long)
    mask = torch.zeros((num_pts,), dtype=torch.bool)
    for idx in indices_list:
        mask[idx.long()] = True
    return torch.nonzero(mask, as_tuple=False).squeeze(1).long()


def approx_spatial_scores(vis_count: torch.Tensor, mean_radii: torch.Tensor, mean_opacity: torch.Tensor) -> torch.Tensor:
    # All tensors are [N]
    return (vis_count.float() + 1e-6) * (mean_radii.float() + 1e-6) * (mean_opacity.float() + 1e-6)


def approx_temporal_scores(velocity_list: List[torch.Tensor]) -> torch.Tensor:
    # velocity_list: list of [N,3] tensors across time samples
    # Compute variance of velocity magnitude as instability proxy, then invert and normalize
    mags = [v.norm(p=2, dim=1) for v in velocity_list]  # each [N]
    mags_stacked = torch.stack(mags, dim=1)  # [N, T]
    var_t = mags_stacked.var(dim=1)  # [N]
    # Normalize to [0,1] with tanh-like mapping: high variance -> small score
    st = 1.0 - torch.tanh(var_t / (var_t.mean() + 1e-6))
    return st.clamp(0.0, 1.0)


def rank_and_prune(ss: torch.Tensor, st: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    s = ss * st
    N = s.numel()
    keep = max(1, int(N * keep_ratio))
    vals, indices = torch.topk(s, k=keep, largest=True)
    keep_mask = torch.zeros((N,), dtype=torch.bool)
    keep_mask[indices] = True
    return keep_mask



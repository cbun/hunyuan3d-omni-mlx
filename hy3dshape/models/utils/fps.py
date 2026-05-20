# -*- coding: utf-8 -*-
"""Farthest-point sampling with an optional torch-cluster fast path."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor


def _sample_count(n_points: int, ratio: Optional[Union[Tensor, float]]) -> int:
    if ratio is None:
        return n_points
    if isinstance(ratio, Tensor):
        ratio = float(ratio.item())
    if ratio <= 0:
        return 0
    if ratio <= 1:
        return max(1, int(round(n_points * ratio)))
    return min(n_points, int(round(ratio)))


def _torch_fps_one(points: Tensor, count: int, random_start: bool) -> Tensor:
    n_points = points.shape[0]
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    count = min(count, n_points)

    selected = torch.empty(count, dtype=torch.long, device=points.device)
    farthest = torch.randint(n_points, (1,), device=points.device).item() if random_start else 0
    min_distances = torch.full((n_points,), float("inf"), device=points.device, dtype=torch.float32)
    points_f32 = points.float()

    for i in range(count):
        selected[i] = farthest
        centroid = points_f32[farthest].unsqueeze(0)
        distances = ((points_f32 - centroid) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        farthest = int(torch.argmax(min_distances).item())

    return selected


def torch_fps(
    src: Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> Tensor:
    if ptr is not None:
        ptr_tensor = torch.as_tensor(ptr, device=src.device, dtype=torch.long)
        outputs = []
        for i in range(len(ptr_tensor) - 1):
            start, end = int(ptr_tensor[i].item()), int(ptr_tensor[i + 1].item())
            local = _torch_fps_one(src[start:end], _sample_count(end - start, ratio), random_start)
            outputs.append(local + start)
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=torch.long, device=src.device)

    if batch is None:
        return _torch_fps_one(src, _sample_count(src.shape[0], ratio), random_start)

    batch = batch.to(src.device)
    if batch_size is None:
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0

    outputs = []
    for batch_idx in range(batch_size):
        indices = torch.nonzero(batch == batch_idx, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        local = _torch_fps_one(src[indices], _sample_count(indices.numel(), ratio), random_start)
        outputs.append(indices[local])

    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dtype=torch.long, device=src.device)


def fps(
    src: Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> Tensor:
    try:
        from torch_cluster import fps as fps_fn
    except ImportError:
        return torch_fps(src, batch, ratio, random_start, batch_size, ptr)
    return fps_fn(src.float(), batch, ratio, random_start, batch_size, ptr)

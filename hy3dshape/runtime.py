# -*- coding: utf-8 -*-
"""Runtime helpers for CUDA, Apple Silicon, and CPU execution."""

from __future__ import annotations

import importlib.util
import os
import platform
from typing import Optional, Union


def prepare_runtime_environment() -> None:
    """Set environment defaults before model execution starts."""
    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


prepare_runtime_environment()

import torch


DeviceLike = Union[str, torch.device, None]


def has_mlx() -> bool:
    return importlib.util.find_spec("mlx") is not None


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        return device

    requested = str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mlx":
        if not has_mlx():
            raise RuntimeError("The 'mlx' package is not installed. Install the Apple Silicon requirements first.")
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MLX was requested, but PyTorch MPS is not available on this machine.")
        return torch.device("mps")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if resolved.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    return resolved


def resolve_dtype(dtype: Union[str, torch.dtype, None] = "auto", device: DeviceLike = "auto") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        dtype = "auto"

    requested = str(dtype).lower()
    if requested == "auto":
        resolved_device = resolve_device(device)
        if resolved_device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested not in aliases:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use auto, fp16, bf16, or fp32.")

    resolved = aliases[requested]
    resolved_device = resolve_device(device)
    if resolved_device.type == "mps" and resolved == torch.bfloat16:
        raise ValueError("MPS does not support bfloat16 for this pipeline; use auto, fp16, or fp32.")
    return resolved


def make_generator(device: DeviceLike, seed: Optional[int] = None) -> torch.Generator:
    resolved_device = resolve_device(device)
    try:
        generator = torch.Generator(device=resolved_device.type)
    except RuntimeError:
        generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def supports_cuda_sdp(tensor_or_device: Union[torch.Tensor, torch.device]) -> bool:
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.device.type == "cuda"
    return torch.device(tensor_or_device).type == "cuda"


def default_geometry_dtype(device: DeviceLike) -> torch.dtype:
    resolved_device = resolve_device(device)
    if resolved_device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def describe_backend(device: DeviceLike, dtype: Union[str, torch.dtype, None] = "auto") -> str:
    native_mlx = not isinstance(device, torch.device) and str(device).lower() == "mlx"
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    backend = resolved_device.type
    if native_mlx:
        backend = "native-mlx+mps"
    return f"{backend} ({resolved_dtype})"

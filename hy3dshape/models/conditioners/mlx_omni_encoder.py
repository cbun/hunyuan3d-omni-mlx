# -*- coding: utf-8 -*-
"""Native MLX control-token projection for Hunyuan3D-Omni conditioning."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

import mlx.core as mx

from hy3dshape.models.denoisers.mlx_hunyuandit import _as_mx, _gelu, _linear, _rms_norm


Array = mx.array


def _dtype_from_name(dtype) -> mx.Dtype:
    if not isinstance(dtype, str):
        return dtype
    return {
        "float16": mx.float16,
        "fp16": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
    }.get(dtype, mx.float16)


def _to_mlx(value, dtype: mx.Dtype) -> Array:
    if isinstance(value, mx.array):
        return value.astype(dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return mx.array(value).astype(dtype)


def _fourier_embed(x: Array, num_freqs: int = 8, include_pi: bool = False) -> Array:
    freqs = 2.0 ** mx.arange(num_freqs, dtype=mx.float32)
    if include_pi:
        freqs = freqs * mx.array(np.pi, dtype=mx.float32)
    embed = mx.reshape(mx.expand_dims(x.astype(mx.float32), -1) * freqs, (*x.shape[:-1], -1))
    out = mx.concatenate([x.astype(mx.float32), mx.sin(embed), mx.cos(embed)], axis=-1)
    return out.astype(x.dtype)


class MLXOmniControlEncoder:
    def __init__(
        self,
        params: Dict[str, Array],
        dtype: mx.Dtype = mx.float16,
        num_freqs: int = 8,
        include_pi: bool = False,
    ):
        self.params = params
        self.dtype = dtype
        self.num_freqs = num_freqs
        self.include_pi = include_pi
        self._compiled_project = None

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint_path: str,
        dtype="float16",
        num_freqs: int = 8,
        include_pi: bool = False,
    ) -> "MLXOmniControlEncoder":
        parsed_dtype = _dtype_from_name(dtype)
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        params = {key: _as_mx(value, parsed_dtype) for key, value in state.items()}
        mx.eval(list(params.values()))
        return cls(params=params, dtype=parsed_dtype, num_freqs=num_freqs, include_pi=include_pi)

    def compile(self) -> "MLXOmniControlEncoder":
        self._compiled_project = mx.compile(lambda control, signal_idx: self._project_control(control, signal_idx))
        return self

    def _p(self, key: str) -> Array:
        return self.params[key]

    def _project_control(self, control: Array, signal_idx: Array) -> Array:
        control = control.astype(self.dtype)
        batch_size = control.shape[0]
        x = _fourier_embed(control, self.num_freqs, self.include_pi)
        x = _linear(x, self._p("linear.0.weight"), self._p("linear.0.bias"))
        x = _rms_norm(x, self._p("linear.1.weight"))
        x = _gelu(x)

        signal = self._p("cond_signal_embedding.weight")[signal_idx]
        signal = _linear(signal, self._p("cond_signal_linear.weight"), self._p("cond_signal_linear.bias"))
        signal = mx.broadcast_to(mx.reshape(signal, (1, 1, -1)), (batch_size, 10, signal.shape[-1]))
        return mx.concatenate([x, signal], axis=1)

    def project_control(self, control: Array, signal_idx: int) -> Array:
        idx = mx.array(signal_idx, dtype=mx.int32)
        if self._compiled_project is not None:
            return self._compiled_project(control, idx)
        return self._project_control(control, idx)

    @staticmethod
    def bbox_to_corners(bbox: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ], dtype=torch.float32, device=bbox.device)
        return bbox / 2 * signs.unsqueeze(0)

    def __call__(
        self,
        image_cond,
        pose: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        point: Optional[torch.Tensor] = None,
        voxel: Optional[torch.Tensor] = None,
        sampled_point: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        image_cond_mx = _to_mlx(image_cond, self.dtype)
        batch_size = image_cond_mx.shape[0]

        if pose is not None:
            control = _to_mlx(pose, self.dtype)
            signal_idx = 0
            sampled = pose[..., :3]
        elif bbox is not None:
            control = _to_mlx(bbox.repeat(1, 1, 2), self.dtype)
            signal_idx = 1
            sampled = self.bbox_to_corners(bbox)
        elif voxel is not None:
            control = _to_mlx(voxel.repeat(1, 1, 2), self.dtype)
            signal_idx = 2
            sampled = voxel[..., :3] if sampled_point is None else sampled_point
        elif point is not None:
            control = _to_mlx(point.repeat(1, 1, 2), self.dtype)
            signal_idx = 3
            sampled = point[..., :3]
        else:
            raise ValueError("pose, bbox, voxel, or point must be provided")

        control_cond = self.project_control(control, signal_idx)
        return {
            "cond": mx.concatenate([image_cond_mx, control_cond], axis=1),
            "cond_point": sampled,
        }


def load_mlx_omni_control_encoder(
    checkpoint_path: str,
    dtype="float16",
    num_freqs: int = 8,
    include_pi: bool = False,
) -> MLXOmniControlEncoder:
    return MLXOmniControlEncoder.from_torch_checkpoint(
        checkpoint_path,
        dtype=dtype,
        num_freqs=num_freqs,
        include_pi=include_pi,
    )

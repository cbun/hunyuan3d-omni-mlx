# -*- coding: utf-8 -*-
"""Native MLX DINOv2 image encoder for inference."""

from __future__ import annotations

from typing import Dict, Union

import gc
import numpy as np
import torch
import torch.nn.functional as F

import mlx.core as mx

from hy3dshape.models.denoisers.mlx_hunyuandit import _as_mx, _attention, _gelu, _layer_norm, _linear


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


class MLXDinoV2Model:
    def __init__(
        self,
        params: Dict[str, Array],
        dtype: mx.Dtype = mx.float16,
        image_size: int = 518,
        patch_size: int = 14,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
    ):
        self.params = params
        self.dtype = dtype
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_size // num_heads
        self._compiled_forward = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "facebook/dinov2-large",
        dtype: Union[str, mx.Dtype] = mx.float16,
        image_size: int = 518,
    ) -> "MLXDinoV2Model":
        from transformers import Dinov2Model

        parsed_dtype = _dtype_from_name(dtype)
        model = Dinov2Model.from_pretrained(model_name).eval()
        state = model.state_dict()
        patch_size = model.config.patch_size
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        params = {key: _as_mx(value, parsed_dtype) for key, value in state.items()}
        cls._pack_params(params, num_layers)
        mx.eval(list(params.values()))
        del state, model
        gc.collect()
        return cls(
            params=params,
            dtype=parsed_dtype,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def compile(self) -> "MLXDinoV2Model":
        self._compiled_forward = mx.compile(self._forward)
        return self

    def release(self) -> None:
        self.params.clear()
        self._compiled_forward = None
        mx.clear_cache()

    @staticmethod
    def _pack_params(params: Dict[str, Array], num_layers: int) -> None:
        patch_weight = "embeddings.patch_embeddings.projection.weight"
        if patch_weight in params:
            params[f"{patch_weight}_nhwc"] = mx.transpose(params[patch_weight], (0, 2, 3, 1))
            del params[patch_weight]

        for layer in range(num_layers):
            prefix = f"encoder.layer.{layer}.attention.attention"
            q_weight = f"{prefix}.query.weight"
            k_weight = f"{prefix}.key.weight"
            v_weight = f"{prefix}.value.weight"
            q_bias = f"{prefix}.query.bias"
            k_bias = f"{prefix}.key.bias"
            v_bias = f"{prefix}.value.bias"
            if not all(key in params for key in (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)):
                continue
            params[f"{prefix}.qkv.weight"] = mx.concatenate(
                [params[q_weight], params[k_weight], params[v_weight]],
                axis=0,
            )
            params[f"{prefix}.qkv.bias"] = mx.concatenate(
                [params[q_bias], params[k_bias], params[v_bias]],
                axis=0,
            )
            for key in (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias):
                del params[key]

    def _p(self, key: str) -> Array:
        return self.params[key]

    def _linear_key(self, x: Array, prefix: str) -> Array:
        return _linear(x, self._p(f"{prefix}.weight"), self._p(f"{prefix}.bias"))

    def _patch_embeddings(self, x: Array) -> Array:
        packed_weight = "embeddings.patch_embeddings.projection.weight_nhwc"
        if packed_weight in self.params:
            weight = self._p(packed_weight)
        else:
            weight = mx.transpose(self._p("embeddings.patch_embeddings.projection.weight"), (0, 2, 3, 1))
        bias = self._p("embeddings.patch_embeddings.projection.bias")
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.conv2d(x, weight, stride=self.patch_size)
        x = x + bias
        return mx.reshape(x, (x.shape[0], -1, self.hidden_size))

    def _embeddings(self, x: Array) -> Array:
        x = self._patch_embeddings(x)
        cls = mx.broadcast_to(self._p("embeddings.cls_token"), (x.shape[0], 1, self.hidden_size))
        x = mx.concatenate([cls, x], axis=1)
        return x + self._p("embeddings.position_embeddings")

    def _attention(self, x: Array, layer: int) -> Array:
        prefix = f"encoder.layer.{layer}.attention"
        bsz, seq_len, _ = x.shape
        qkv_prefix = f"{prefix}.attention.qkv"
        if f"{qkv_prefix}.weight" in self.params:
            q, k, v = mx.split(self._linear_key(x, qkv_prefix), 3, axis=-1)
        else:
            q = self._linear_key(x, f"{prefix}.attention.query")
            k = self._linear_key(x, f"{prefix}.attention.key")
            v = self._linear_key(x, f"{prefix}.attention.value")
        q = mx.reshape(q, (bsz, seq_len, self.num_heads, self.head_dim))
        k = mx.reshape(k, (bsz, seq_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (bsz, seq_len, self.num_heads, self.head_dim))
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = _attention(q, k, v, scale=self.head_dim ** -0.5)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (bsz, seq_len, self.hidden_size))
        return self._linear_key(out, f"{prefix}.output.dense")

    def _mlp(self, x: Array, layer: int) -> Array:
        prefix = f"encoder.layer.{layer}.mlp"
        x = self._linear_key(x, f"{prefix}.fc1")
        x = _gelu(x)
        return self._linear_key(x, f"{prefix}.fc2")

    def _layer(self, x: Array, layer: int) -> Array:
        prefix = f"encoder.layer.{layer}"
        residual = self._attention(
            _layer_norm(x, self._p(f"{prefix}.norm1.weight"), self._p(f"{prefix}.norm1.bias")),
            layer,
        )
        x = x + residual * self._p(f"{prefix}.layer_scale1.lambda1")
        residual = self._mlp(
            _layer_norm(x, self._p(f"{prefix}.norm2.weight"), self._p(f"{prefix}.norm2.bias")),
            layer,
        )
        return x + residual * self._p(f"{prefix}.layer_scale2.lambda1")

    def _forward(self, pixel_values: Array) -> Array:
        x = pixel_values.astype(self.dtype)
        x = self._embeddings(x)
        for layer in range(self.num_layers):
            x = self._layer(x, layer)
        return _layer_norm(x, self._p("layernorm.weight"), self._p("layernorm.bias"))

    @staticmethod
    def preprocess_torch_image(image: torch.Tensor, image_size: int = 518) -> torch.Tensor:
        image = (image + 1.0) / 2.0
        if image.shape[-2:] != (image_size, image_size):
            image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image - mean) / std

    def __call__(self, image) -> Array:
        if hasattr(image, "detach"):
            image = self.preprocess_torch_image(image, self.image_size)
            image = mx.array(image.detach().cpu().numpy()).astype(self.dtype)
        if self._compiled_forward is not None:
            return self._compiled_forward(image)
        return self._forward(image)


def load_mlx_dinov2(
    model_name: str = "facebook/dinov2-large",
    dtype: Union[str, mx.Dtype] = "float16",
    image_size: int = 518,
) -> MLXDinoV2Model:
    return MLXDinoV2Model.from_pretrained(model_name, dtype=dtype, image_size=image_size)

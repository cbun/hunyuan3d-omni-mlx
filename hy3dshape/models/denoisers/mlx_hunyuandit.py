# -*- coding: utf-8 -*-
"""Native MLX implementation of the Hunyuan3D-Omni denoiser.

This module mirrors ``hunyuandit.HunYuanDiTPlain`` for inference and can load
the original PyTorch checkpoint directly. It is intentionally functional rather
than ``mlx.nn.Module``-heavy so the PyTorch state-dict key names stay stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

import mlx.core as mx


Array = mx.array


def _as_mx(value, dtype: mx.Dtype = mx.float16) -> Array:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = mx.array(value)
    if arr.dtype != dtype and arr.dtype in (mx.float16, mx.float32, mx.bfloat16):
        arr = arr.astype(dtype)
    return arr


def _gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _silu(x: Array) -> Array:
    return x * mx.sigmoid(x)


def _linear(x: Array, weight: Array, bias: Optional[Array] = None) -> Array:
    y = mx.matmul(x, mx.swapaxes(weight, -1, -2))
    if bias is not None:
        y = y + bias
    return y


def _layer_norm(x: Array, weight: Array, bias: Array, eps: float = 1e-6) -> Array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    y = (x - mx.mean(x, axis=-1, keepdims=True))
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + eps)
    y = y * weight.astype(mx.float32) + bias.astype(mx.float32)
    return y.astype(dtype)


def _rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    y = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    y = y * weight.astype(mx.float32)
    return y.astype(dtype)


def _timesteps(timesteps: Array, num_channels: int, max_period: int = 10000) -> Array:
    half_dim = num_channels // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / half_dim
    emb = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if num_channels % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def _attention(q: Array, k: Array, v: Array, scale: float) -> Array:
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def _topk(values: Array, k: int, axis: int = -1) -> tuple[Array, Array]:
    indices = mx.argpartition(values, kth=values.shape[axis] - k, axis=axis)
    topk_idx = mx.take(indices, mx.arange(values.shape[axis] - k, values.shape[axis]), axis=axis)
    topk_values = mx.take_along_axis(values, topk_idx, axis=axis)
    return topk_values, topk_idx


@dataclass(frozen=True)
class MLXHunYuanDiTConfig:
    input_size: int = 4096
    in_channels: int = 64
    hidden_size: int = 2048
    context_dim: int = 1024
    depth: int = 21
    num_heads: int = 16
    qk_norm: bool = True
    text_len: int = 1370
    with_decoupled_ca: bool = False
    use_attention_pooling: bool = False
    use_pos_emb: bool = False
    qkv_bias: bool = False
    num_moe_layers: int = 6
    num_experts: int = 8
    moe_top_k: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class MLXHunYuanDiTPlain:
    def __init__(
        self,
        params: Dict[str, Array],
        config: MLXHunYuanDiTConfig = MLXHunYuanDiTConfig(),
        dtype: mx.Dtype = mx.float16,
        sparse_moe: bool = False,
    ):
        self.params = params
        self.config = config
        self.dtype = dtype
        self.sparse_moe = sparse_moe
        self._compiled_forward = None

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Union[MLXHunYuanDiTConfig, dict]] = None,
        dtype: mx.Dtype = mx.float16,
        sparse_moe: bool = False,
    ) -> "MLXHunYuanDiTPlain":
        import torch

        if config is None:
            parsed_config = MLXHunYuanDiTConfig()
        elif isinstance(config, MLXHunYuanDiTConfig):
            parsed_config = config
        else:
            allowed = {field.name for field in MLXHunYuanDiTConfig.__dataclass_fields__.values()}
            parsed_config = MLXHunYuanDiTConfig(**{k: v for k, v in config.items() if k in allowed})

        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        params = {key: _as_mx(value, dtype=dtype) for key, value in state.items()}
        mx.eval(list(params.values()))
        return cls(params=params, config=parsed_config, dtype=dtype, sparse_moe=sparse_moe)

    def _p(self, key: str) -> Array:
        return self.params[key]

    def _linear_key(self, x: Array, prefix: str, bias: bool = True) -> Array:
        return _linear(x, self._p(f"{prefix}.weight"), self._p(f"{prefix}.bias") if bias and f"{prefix}.bias" in self.params else None)

    def _time_embedder(self, t: Array, guidance_cond: Optional[Array] = None) -> Array:
        t_freq = _timesteps(t, self.config.hidden_size).astype(self.dtype)
        if guidance_cond is not None and "t_embedder.cond_proj.weight" in self.params:
            t_freq = t_freq + self._linear_key(guidance_cond, "t_embedder.cond_proj", bias=False)
        x = self._linear_key(t_freq, "t_embedder.mlp.0")
        x = _gelu(x)
        x = self._linear_key(x, "t_embedder.mlp.2")
        return mx.expand_dims(x, axis=1)

    def _self_attention(self, x: Array, prefix: str) -> Array:
        cfg = self.config
        bsz, seq_len, _ = x.shape

        q = self._linear_key(x, f"{prefix}.to_q", bias=cfg.qkv_bias)
        k = self._linear_key(x, f"{prefix}.to_k", bias=cfg.qkv_bias)
        v = self._linear_key(x, f"{prefix}.to_v", bias=cfg.qkv_bias)

        qkv = mx.concatenate([q, k, v], axis=-1)
        qkv = mx.reshape(qkv, (1, -1, cfg.num_heads, cfg.head_dim * 3))
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = mx.reshape(q, (bsz, seq_len, cfg.num_heads, cfg.head_dim))
        k = mx.reshape(k, (bsz, seq_len, cfg.num_heads, cfg.head_dim))
        v = mx.reshape(v, (bsz, seq_len, cfg.num_heads, cfg.head_dim))

        if cfg.qk_norm:
            q = _rms_norm(q, self._p(f"{prefix}.q_norm.weight"))
            k = _rms_norm(k, self._p(f"{prefix}.k_norm.weight"))

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = _attention(q, k, v, scale=cfg.head_dim ** -0.5)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (bsz, seq_len, cfg.hidden_size))
        return self._linear_key(out, f"{prefix}.out_proj")

    def _cross_attention(self, x: Array, cond: Array, prefix: str) -> Array:
        cfg = self.config
        bsz, x_len, _ = x.shape
        cond_len = cond.shape[1]

        q = self._linear_key(x, f"{prefix}.to_q", bias=cfg.qkv_bias)
        k = self._linear_key(cond, f"{prefix}.to_k", bias=cfg.qkv_bias)
        v = self._linear_key(cond, f"{prefix}.to_v", bias=cfg.qkv_bias)

        kv = mx.concatenate([k, v], axis=-1)
        kv = mx.reshape(kv, (1, -1, cfg.num_heads, cfg.head_dim * 2))
        k, v = mx.split(kv, 2, axis=-1)

        q = mx.reshape(q, (bsz, x_len, cfg.num_heads, cfg.head_dim))
        k = mx.reshape(k, (bsz, cond_len, cfg.num_heads, cfg.head_dim))
        v = mx.reshape(v, (bsz, cond_len, cfg.num_heads, cfg.head_dim))

        if cfg.qk_norm:
            q = _rms_norm(q, self._p(f"{prefix}.q_norm.weight"))
            k = _rms_norm(k, self._p(f"{prefix}.k_norm.weight"))

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = _attention(q, k, v, scale=cfg.head_dim ** -0.5)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (bsz, x_len, cfg.hidden_size))
        return self._linear_key(out, f"{prefix}.out_proj")

    def _mlp(self, x: Array, prefix: str) -> Array:
        x = self._linear_key(x, f"{prefix}.fc1")
        x = _gelu(x)
        return self._linear_key(x, f"{prefix}.fc2")

    def _feed_forward(self, x: Array, prefix: str) -> Array:
        x = self._linear_key(x, f"{prefix}.net.0.proj")
        x = _gelu(x)
        return self._linear_key(x, f"{prefix}.net.2")

    def _moe_dense(self, x: Array, prefix: str) -> Array:
        cfg = self.config
        orig_shape = x.shape
        flat = mx.reshape(x, (-1, orig_shape[-1]))

        logits = _linear(flat, self._p(f"{prefix}.gate.weight"), None)
        scores = mx.softmax(logits, axis=-1)
        topk_weight, topk_idx = _topk(scores, k=cfg.moe_top_k, axis=-1)

        y = mx.zeros_like(flat)
        for expert_idx in range(cfg.num_experts):
            expert_out = self._feed_forward(flat, f"{prefix}.experts.{expert_idx}")
            mask = (topk_idx == expert_idx).astype(expert_out.dtype)
            weights = mx.sum(topk_weight * mask, axis=-1, keepdims=True)
            y = y + expert_out * weights

        y = y + self._feed_forward(mx.reshape(x, (-1, orig_shape[-1])), f"{prefix}.shared_experts")
        return mx.reshape(y, orig_shape)

    def _moe_sparse(self, x: Array, prefix: str) -> Array:
        cfg = self.config
        orig_shape = x.shape
        flat = mx.reshape(x, (-1, orig_shape[-1]))

        logits = _linear(flat, self._p(f"{prefix}.gate.weight"), None)
        scores = mx.softmax(logits, axis=-1)
        topk_weight, topk_idx = _topk(scores, k=cfg.moe_top_k, axis=-1)
        mx.eval(topk_weight, topk_idx)

        topk_idx_np = np.array(topk_idx)
        topk_weight_np = np.array(topk_weight, dtype=np.float32)
        y = mx.zeros_like(flat)
        for expert_idx in range(cfg.num_experts):
            token_rows, token_slots = np.where(topk_idx_np == expert_idx)
            if len(token_rows) == 0:
                continue
            row_idx = mx.array(token_rows.astype(np.int32))
            expert_tokens = mx.take(flat, row_idx, axis=0)
            weights = mx.array(topk_weight_np[token_rows, token_slots]).astype(expert_tokens.dtype)
            expert_out = self._feed_forward(expert_tokens, f"{prefix}.experts.{expert_idx}")
            expert_out = expert_out * mx.expand_dims(weights, axis=-1)
            current = mx.take(y, row_idx, axis=0)
            y = mx.put_along_axis(y, mx.expand_dims(row_idx, axis=-1), current + expert_out, axis=0)

        y = y + self._feed_forward(mx.reshape(x, (-1, orig_shape[-1])), f"{prefix}.shared_experts")
        return mx.reshape(y, orig_shape)

    def _moe(self, x: Array, prefix: str) -> Array:
        if self.sparse_moe:
            return self._moe_sparse(x, prefix)
        return self._moe_dense(x, prefix)

    def _block(self, x: Array, cond: Array, layer: int, skip_value: Optional[Array] = None) -> Array:
        prefix = f"blocks.{layer}"
        if layer > self.config.depth // 2:
            x = mx.concatenate([skip_value, x], axis=-1)
            x = self._linear_key(x, f"{prefix}.skip_linear")
            x = _layer_norm(x, self._p(f"{prefix}.skip_norm.weight"), self._p(f"{prefix}.skip_norm.bias"))

        residual = self._self_attention(
            _layer_norm(x, self._p(f"{prefix}.norm1.weight"), self._p(f"{prefix}.norm1.bias")),
            f"{prefix}.attn1",
        )
        x = x + residual

        residual = self._cross_attention(
            _layer_norm(x, self._p(f"{prefix}.norm2.weight"), self._p(f"{prefix}.norm2.bias")),
            cond,
            f"{prefix}.attn2",
        )
        x = x + residual

        mlp_inputs = _layer_norm(x, self._p(f"{prefix}.norm3.weight"), self._p(f"{prefix}.norm3.bias"))
        if self.config.depth - layer <= self.config.num_moe_layers:
            x = x + self._moe(mlp_inputs, f"{prefix}.moe")
        else:
            x = x + self._mlp(mlp_inputs, f"{prefix}.mlp")
        return x

    def compile(self) -> "MLXHunYuanDiTPlain":
        if self.sparse_moe:
            return self
        self._compiled_forward = mx.compile(lambda x, t, cond: self._forward(x, t, cond, None))
        return self

    def _forward(self, x: Array, t: Array, cond: Array, guidance_cond: Optional[Array] = None) -> Array:
        x = x.astype(self.dtype)
        t = t.astype(self.dtype)
        cond = cond.astype(self.dtype)

        c = self._time_embedder(t, guidance_cond=guidance_cond)
        x = self._linear_key(x, "x_embedder")
        x = mx.concatenate([c, x], axis=1)

        skip_values = []
        for layer in range(self.config.depth):
            skip_value = None if layer <= self.config.depth // 2 else skip_values.pop()
            x = self._block(x, cond, layer, skip_value=skip_value)
            if layer < self.config.depth // 2:
                skip_values.append(x)

        x = _layer_norm(x, self._p("final_layer.norm_final.weight"), self._p("final_layer.norm_final.bias"))
        x = x[:, 1:]
        x = self._linear_key(x, "final_layer.linear")
        return x

    def __call__(self, x: Array, t: Array, contexts: Dict[str, Array], guidance_cond: Optional[Array] = None) -> Array:
        cond = contexts.get("main", contexts.get("cond"))
        if cond is None:
            raise KeyError("contexts must contain 'main' or 'cond'")
        if guidance_cond is None and self._compiled_forward is not None:
            return self._compiled_forward(x, t, cond)
        return self._forward(x, t, cond, guidance_cond=guidance_cond)


def load_mlx_hunyuan_dit_plain(
    checkpoint_path: str,
    config: Optional[dict] = None,
    dtype: str = "float16",
    sparse_moe: bool = False,
) -> MLXHunYuanDiTPlain:
    dtype_map = {
        "float16": mx.float16,
        "fp16": mx.float16,
        "float32": mx.float32,
        "fp32": mx.float32,
        "bfloat16": mx.bfloat16,
        "bf16": mx.bfloat16,
    }
    return MLXHunYuanDiTPlain.from_torch_checkpoint(
        checkpoint_path,
        config=config,
        dtype=dtype_map.get(dtype, mx.float16),
        sparse_moe=sparse_moe,
    )

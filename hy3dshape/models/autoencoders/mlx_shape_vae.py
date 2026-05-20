# -*- coding: utf-8 -*-
"""Native MLX inference path for the Hunyuan3D ShapeVAE decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from skimage import measure
from tqdm import tqdm

import mlx.core as mx

from hy3dshape.models.denoisers.mlx_hunyuandit import _as_mx, _attention, _gelu, _layer_norm, _linear


Array = mx.array


class MLXLatent2MeshOutput:
    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None


@dataclass(frozen=True)
class MLXShapeVAEConfig:
    num_latents: int = 4096
    embed_dim: int = 64
    width: int = 1024
    heads: int = 16
    num_decoder_layers: int = 16
    num_freqs: int = 8
    include_pi: bool = False

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def latent_shape(self) -> Tuple[int, int]:
        return (self.num_latents, self.embed_dim)


def _dtype_from_name(dtype: Union[str, mx.Dtype]) -> mx.Dtype:
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


def _fourier_embed(x: Array, num_freqs: int, include_pi: bool) -> Array:
    freqs = 2.0 ** mx.arange(num_freqs, dtype=mx.float32)
    if include_pi:
        freqs = freqs * mx.array(np.pi, dtype=mx.float32)
    embed = mx.reshape(mx.expand_dims(x.astype(mx.float32), -1) * freqs, (*x.shape[:-1], -1))
    out = mx.concatenate([x.astype(mx.float32), mx.sin(embed), mx.cos(embed)], axis=-1)
    return out.astype(x.dtype)


class MLXShapeVAEDecoder:
    def __init__(
        self,
        params: Dict[str, Array],
        config: MLXShapeVAEConfig = MLXShapeVAEConfig(),
        dtype: mx.Dtype = mx.float16,
    ):
        self.params = params
        self.config = config
        self.dtype = dtype
        self.fast_decode = False
        self._compiled_decode = None
        self._compiled_query = None

    @property
    def latent_shape(self) -> Tuple[int, int]:
        return self.config.latent_shape

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Union[MLXShapeVAEConfig, dict]] = None,
        dtype: Union[str, mx.Dtype] = mx.float16,
    ) -> "MLXShapeVAEDecoder":
        import torch

        parsed_dtype = _dtype_from_name(dtype)
        if config is None:
            parsed_config = MLXShapeVAEConfig()
        elif isinstance(config, MLXShapeVAEConfig):
            parsed_config = config
        else:
            allowed = {field.name for field in MLXShapeVAEConfig.__dataclass_fields__.values()}
            parsed_config = MLXShapeVAEConfig(**{k: v for k, v in config.items() if k in allowed})

        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        params = {
            key: _as_mx(value, dtype=parsed_dtype)
            for key, value in state.items()
            if key.startswith(("sal.post_kl.", "sal.transformer.", "sal.geo_decoder."))
        }
        mx.eval(list(params.values()))
        return cls(params=params, config=parsed_config, dtype=parsed_dtype)

    def compile(self) -> "MLXShapeVAEDecoder":
        self._compiled_decode = mx.compile(self._decode_latents)
        self._compiled_query = mx.compile(self._query_geometry)
        return self

    def _p(self, key: str) -> Array:
        return self.params[key]

    def _linear_key(self, x: Array, prefix: str, bias: bool = True) -> Array:
        bias_value = self._p(f"{prefix}.bias") if bias and f"{prefix}.bias" in self.params else None
        return _linear(x, self._p(f"{prefix}.weight"), bias_value)

    def _qkv_self_attention(self, x: Array, prefix: str) -> Array:
        cfg = self.config
        bsz, seq_len, _ = x.shape
        qkv = self._linear_key(x, f"{prefix}.attn.c_qkv", bias=False)
        qkv = mx.reshape(qkv, (bsz, seq_len, cfg.heads, cfg.head_dim * 3))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = _layer_norm(
            q,
            self._p(f"{prefix}.attn.attention.q_norm.weight"),
            self._p(f"{prefix}.attn.attention.q_norm.bias"),
        )
        k = _layer_norm(
            k,
            self._p(f"{prefix}.attn.attention.k_norm.weight"),
            self._p(f"{prefix}.attn.attention.k_norm.bias"),
        )
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = _attention(q, k, v, scale=cfg.head_dim ** -0.5)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (bsz, seq_len, cfg.width))
        return self._linear_key(out, f"{prefix}.attn.c_proj")

    def _cross_attention(self, x: Array, data: Array, prefix: str) -> Array:
        cfg = self.config
        bsz, x_len, _ = x.shape
        data_len = data.shape[1]
        q = self._linear_key(x, f"{prefix}.attn.c_q", bias=False)
        kv = self._linear_key(data, f"{prefix}.attn.c_kv", bias=False)
        q = mx.reshape(q, (bsz, x_len, cfg.heads, cfg.head_dim))
        kv = mx.reshape(kv, (bsz, data_len, cfg.heads, cfg.head_dim * 2))
        k, v = mx.split(kv, 2, axis=-1)
        q = _layer_norm(
            q,
            self._p(f"{prefix}.attn.attention.q_norm.weight"),
            self._p(f"{prefix}.attn.attention.q_norm.bias"),
        )
        k = _layer_norm(
            k,
            self._p(f"{prefix}.attn.attention.k_norm.weight"),
            self._p(f"{prefix}.attn.attention.k_norm.bias"),
        )
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = _attention(q, k, v, scale=cfg.head_dim ** -0.5)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (bsz, x_len, cfg.width))
        return self._linear_key(out, f"{prefix}.attn.c_proj")

    def _mlp(self, x: Array, prefix: str) -> Array:
        x = self._linear_key(x, f"{prefix}.mlp.c_fc")
        x = _gelu(x)
        return self._linear_key(x, f"{prefix}.mlp.c_proj")

    def _self_block(self, x: Array, layer: int) -> Array:
        prefix = f"sal.transformer.resblocks.{layer}"
        x = x + self._qkv_self_attention(
            _layer_norm(x, self._p(f"{prefix}.ln_1.weight"), self._p(f"{prefix}.ln_1.bias")),
            prefix,
        )
        x = x + self._mlp(
            _layer_norm(x, self._p(f"{prefix}.ln_2.weight"), self._p(f"{prefix}.ln_2.bias")),
            prefix,
        )
        return x

    def _cross_block(self, x: Array, data: Array, prefix: str) -> Array:
        x = x + self._cross_attention(
            _layer_norm(x, self._p(f"{prefix}.ln_1.weight"), self._p(f"{prefix}.ln_1.bias")),
            _layer_norm(data, self._p(f"{prefix}.ln_2.weight"), self._p(f"{prefix}.ln_2.bias")),
            prefix,
        )
        x = x + self._mlp(
            _layer_norm(x, self._p(f"{prefix}.ln_3.weight"), self._p(f"{prefix}.ln_3.bias")),
            prefix,
        )
        return x

    def _decode_latents(self, latents: Array) -> Array:
        latents = latents.astype(self.dtype)
        x = self._linear_key(latents, "sal.post_kl")
        for layer in range(self.config.num_decoder_layers):
            x = self._self_block(x, layer)
        return x

    def decode_latents(self, latents: Array) -> Array:
        if self._compiled_decode is not None:
            return self._compiled_decode(latents)
        return self._decode_latents(latents)

    def _query_geometry(self, queries: Array, latents: Array) -> Array:
        queries = queries.astype(self.dtype)
        latents = latents.astype(self.dtype)
        x = self._linear_key(
            _fourier_embed(queries, self.config.num_freqs, self.config.include_pi),
            "sal.geo_decoder.query_proj",
        )
        x = self._cross_block(x, latents, "sal.geo_decoder.cross_attn_decoder")
        x = _layer_norm(x, self._p("sal.geo_decoder.ln_post.weight"), self._p("sal.geo_decoder.ln_post.bias"))
        x = self._linear_key(x, "sal.geo_decoder.output_proj")
        return mx.squeeze(x, axis=-1)

    def query_geometry(self, queries: Array, latents: Array) -> Array:
        if self._compiled_query is not None:
            return self._compiled_query(queries, latents)
        return self._query_geometry(queries, latents)

    @staticmethod
    def _dense_grid(
        bounds: Union[Tuple[float], Sequence[float], float],
        octree_depth: int,
        octree_resolution: Optional[int],
    ) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3], dtype=np.float32)
        bbox_max = np.array(bounds[3:6], dtype=np.float32)
        bbox_size = bbox_max - bbox_min
        num_cells = int(octree_resolution or np.exp2(octree_depth))
        x = np.linspace(bbox_min[0], bbox_max[0], num_cells + 1, dtype=np.float32)
        y = np.linspace(bbox_min[1], bbox_max[1], num_cells + 1, dtype=np.float32)
        z = np.linspace(bbox_min[2], bbox_max[2], num_cells + 1, dtype=np.float32)
        xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")
        xyz = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        return xyz, [num_cells + 1, num_cells + 1, num_cells + 1], bbox_size, bbox_min

    def latent2mesh(
        self,
        decoded_latents: Array,
        bounds: Union[Tuple[float], Sequence[float], float] = 1.01,
        octree_depth: int = 8,
        num_chunks: int = 250000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        mc_mode: str = "mc",
        disable_tqdm: bool = True,
    ) -> List[Optional[MLXLatent2MeshOutput]]:
        if mc_mode != "mc":
            raise ValueError("Native MLX ShapeVAE currently supports mc_mode='mc'.")
        if mc_level == -1:
            print("Training with soft labels, inference with sigmoid and marching cubes level 0.")
        elif mc_level == 0:
            print("VAE Trained with TSDF, inference with marching cubes level 0.")
        else:
            print(f"VAE Trained with Occupancy, inference with marching cubes level {mc_level}.")

        xyz_samples, grid_size, bbox_size, bbox_min = self._dense_grid(bounds, octree_depth, octree_resolution)
        batch_size = decoded_latents.shape[0]
        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"MLX MC Level {mc_level} Implicit Function:",
            disable=disable_tqdm,
            leave=False,
        ):
            queries = mx.array(xyz_samples[start:start + num_chunks, :]).astype(self.dtype)
            queries = mx.broadcast_to(mx.expand_dims(queries, axis=0), (batch_size, queries.shape[0], queries.shape[1]))
            logits = self.query_geometry(queries, decoded_latents)
            if mc_level == -1:
                mc_level = 0
                logits = mx.sigmoid(logits) * 2 - 1
            mx.eval(logits)
            batch_logits.append(np.array(logits, dtype=np.float32))

        grid_logits = np.concatenate(batch_logits, axis=1).reshape((batch_size, *grid_size))
        outputs = []
        for batch_idx in range(batch_size):
            try:
                vertices, faces, _, _ = measure.marching_cubes(grid_logits[batch_idx], mc_level, method="lewiner")
                vertices = vertices / (np.array(grid_size) - 1) * bbox_size + bbox_min
            except (ValueError, RuntimeError) as exc:
                print(f"Warning: marching cubes did not produce a valid surface: {exc}")
                outputs.append(None)
                continue
            out = MLXLatent2MeshOutput()
            out.mesh_v = vertices.astype(np.float32)
            out.mesh_f = np.ascontiguousarray(faces)
            outputs.append(out)
        return outputs

    def decode(
        self,
        z_q,
        bounds: Union[Tuple[float], Sequence[float], float] = 1.01,
        octree_depth: int = 8,
        num_chunks: int = 250000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        mc_mode: str = "mc",
        **kwargs,
    ) -> List[Optional[MLXLatent2MeshOutput]]:
        if hasattr(z_q, "detach"):
            z_q = mx.array(z_q.detach().cpu().numpy()).astype(self.dtype)
        decoded = self.decode_latents(z_q)
        mx.eval(decoded)
        return self.latent2mesh(
            decoded,
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            mc_level=mc_level,
            octree_resolution=octree_resolution,
            mc_mode=mc_mode,
            disable_tqdm=True,
        )


def load_mlx_shape_vae_decoder(
    checkpoint_path: str,
    config: Optional[dict] = None,
    dtype: Union[str, mx.Dtype] = "float16",
) -> MLXShapeVAEDecoder:
    return MLXShapeVAEDecoder.from_torch_checkpoint(checkpoint_path, config=config, dtype=dtype)

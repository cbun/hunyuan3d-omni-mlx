# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from typing import List, Optional, Union
import gc
import os
import json
import numpy as np
import trimesh
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from huggingface_hub import snapshot_download
from diffusers.utils.torch_utils import randn_tensor

from .utils import init_from_ckpt, export_to_trimesh, synchronize_timer
from hy3dshape.models.utils.misc import get_config_from_file, instantiate_from_config
from hy3dshape.runtime import resolve_device, resolve_dtype


def _torch_dtype_name(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported dtype for MLX conversion: {dtype}")


def _torch_dtype_to_mlx(dtype):
    import mlx.core as mx

    if dtype == torch.float16:
        return mx.float16
    if dtype == torch.float32:
        return mx.float32
    if dtype == torch.bfloat16:
        return mx.bfloat16
    raise ValueError(f"Unsupported dtype for MLX conversion: {dtype}")


def _torch_tensor_to_mlx(tensor, dtype):
    import mlx.core as mx

    array = mx.array(tensor.detach().cpu().numpy())
    return array.astype(_torch_dtype_to_mlx(dtype))


def _mlx_array_to_torch(array, device, dtype):
    import mlx.core as mx

    mx.eval(array)
    return torch.from_numpy(np.array(array)).to(device=device, dtype=dtype)


class Hunyuan3DOmniSiTFlowMatchingPipeline:
    '''
    This pipeline is designed for generating 3D shapes using the Hunyuan3DOmni model with SiT flow matching.
    '''
    @classmethod
    def from_pretrained(cls,
                        model_path,
                        variant=None,
                        device='auto',
                        dtype='auto',
                        resume_download=False,
                        force_download=False,
                        revision="main",
                        **kwargs):

        native_mlx = str(device).lower() == "mlx" or bool(kwargs.pop("native_mlx", False))
        device = resolve_device(device)
        dtype = resolve_dtype(dtype, device)

        if os.path.exists(model_path):
            print(f'Loading model from local path: {model_path}')
        else:
            repo_id = model_path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, repo_id))
            print(f'Loading model from huggingface cache: {model_path}')
            if not os.path.exists(model_path):
                print(f'Not Found {model_path}')
                print(f'Downloading model from huggingface: {repo_id}')

            path = snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=resume_download,
                force_download=force_download, 
                revision=revision,
            )
            print('path', path)

        ckpt_name = 'pytorch_model.bin'
        submodules = dict()
        for submodule_name in ['model', 'vae','cond_encoder', 'scheduler', 'image_processor']:
            config_path = os.path.join(model_path, submodule_name, 'config.json')
            config = get_config_from_file(config_path)

            if submodule_name == "scheduler":
                if 'transport' in config and 'sampler' in config:
                    transport = instantiate_from_config(config.transport)
                    submodule = instantiate_from_config(config.sampler, transport=transport)
                else:
                    scheduler_config = kwargs.get('scheduler_config', config.scheduler_cfg.get('denoise'))
                    submodule = instantiate_from_config(scheduler_config)
                submodules[submodule_name] = submodule
                print(f'Loaded {submodule_name}')
                continue

            if submodule_name == "model" and native_mlx:
                from hy3dshape.models.denoisers.mlx_hunyuandit import load_mlx_hunyuan_dit_plain

                ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if variant == 'ema':
                    ema_ckpt_path = os.path.join(model_path, submodule_name, 'pytorch_model_ema.bin')
                    if os.path.exists(ema_ckpt_path):
                        ckpt_path = ema_ckpt_path
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Native MLX model checkpoint not found: {ckpt_path}")
                submodule = load_mlx_hunyuan_dit_plain(
                    ckpt_path,
                    config=dict(config.get("params", {})),
                    dtype=_torch_dtype_name(dtype),
                    sparse_moe=bool(kwargs.get("sparse_mlx_moe", False)),
                )
                if kwargs.get("compile_mlx", True) and hasattr(submodule, "compile"):
                    submodule.compile()
                submodules[submodule_name] = submodule
                print(f'Loaded {submodule_name} with native MLX from {ckpt_path}')
                continue

            if submodule_name == "vae" and native_mlx:
                from hy3dshape.models.autoencoders.mlx_shape_vae import load_mlx_shape_vae_decoder

                ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Native MLX VAE checkpoint not found: {ckpt_path}")
                module_params = dict(config.params.module_cfg.params)
                submodule = load_mlx_shape_vae_decoder(
                    ckpt_path,
                    config=module_params,
                    dtype=_torch_dtype_name(dtype),
                )
                if kwargs.get("compile_mlx", True) and hasattr(submodule, "compile"):
                    submodule.compile()
                submodules[submodule_name] = submodule
                print(f'Loaded {submodule_name} with native MLX from {ckpt_path}')
                continue

            if submodule_name == "cond_encoder" and native_mlx and bool(kwargs.get("native_mlx_dino", False)):
                from hy3dshape.models.conditioners.mlx_dinov2 import load_mlx_dinov2
                from hy3dshape.models.conditioners.mlx_omni_encoder import load_mlx_omni_control_encoder

                cond_params = dict(config.get("params", {}))
                image_params = dict(cond_params.get("image_encoder", {}).get("params", {}))
                ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                mlx_dino_encoder = load_mlx_dinov2(
                    image_params.get("version", "facebook/dinov2-large"),
                    dtype=_torch_dtype_name(dtype),
                    image_size=image_params.get("image_size", 518),
                )
                mlx_control_encoder = load_mlx_omni_control_encoder(
                    ckpt_path,
                    dtype=_torch_dtype_name(dtype),
                    num_freqs=cond_params.get("num_freqs", 8),
                    include_pi=cond_params.get("include_pi", False),
                )
                if kwargs.get("compile_mlx_dino", False) and hasattr(mlx_dino_encoder, "compile"):
                    mlx_dino_encoder.compile()
                if kwargs.get("compile_mlx", True) and hasattr(mlx_control_encoder, "compile"):
                    mlx_control_encoder.compile()
                submodules[submodule_name] = None
                submodules["mlx_dino_encoder"] = mlx_dino_encoder
                submodules["mlx_control_encoder"] = mlx_control_encoder
                print(f'Loaded {submodule_name} with native MLX DINO and control projection')
                continue

            submodule = instantiate_from_config(config)
            if isinstance(submodule, nn.Module):
                ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if variant == 'ema':
                    ckpt_name = 'pytorch_model_ema.bin'
                    ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                    if not os.path.exists(ckpt_path):
                        ckpt_name = 'pytorch_model.bin'
                        ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if os.path.exists(ckpt_path):
                    missing, unexpected = submodule.load_state_dict(
                        torch.load(ckpt_path, map_location='cpu'), strict=False)
                    print(f"Loaded {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
                    if len(missing) > 0:
                        print(f"Missing Keys: {Counter([s.split('.')[0] for s in missing])}")
                    if len(unexpected) > 0:
                        print(f"Unexpected Keys: {Counter([s.split('.')[0] for s in unexpected])}")

                submodule.disable_drop = True
                submodule = submodule.to(device=device, dtype=dtype)
                submodule.eval()
            submodules[submodule_name] = submodule
            if submodule_name == "cond_encoder" and native_mlx and bool(kwargs.get("native_mlx_control", False)):
                from hy3dshape.models.conditioners.mlx_omni_encoder import load_mlx_omni_control_encoder

                cond_params = dict(config.get("params", {}))
                mlx_control_encoder = load_mlx_omni_control_encoder(
                    ckpt_path,
                    dtype=_torch_dtype_name(dtype),
                    num_freqs=cond_params.get("num_freqs", 8),
                    include_pi=cond_params.get("include_pi", False),
                )
                if kwargs.get("compile_mlx", True) and hasattr(mlx_control_encoder, "compile"):
                    mlx_control_encoder.compile()
                submodules["mlx_control_encoder"] = mlx_control_encoder
                print(f'Loaded {submodule_name} control projection with native MLX from {ckpt_path}')
            print(f'Loaded {submodule_name}')

        vae_config_path = os.path.join(model_path, 'vae', 'config.json')
        vae_config = get_config_from_file(vae_config_path)
        model_kwargs = dict(
            scale_factor=vae_config['scale_factor'],
            device=device,
            dtype=dtype,
            native_mlx=native_mlx,
        )
        model_kwargs.update(submodules)
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    def __init__(self,
                vae=None,
                model=None,
                scheduler=None,
                cond_encoder=None,
                mlx_dino_encoder=None,
                mlx_control_encoder=None,
                image_processor=None,
                scale_factor=1.0,
                device='auto',
                dtype='auto',
                **kwargs):
        self.vae = vae                           # Shape-VAE
        self.model = model                       # Shape-DiT
        self.scheduler = scheduler               # Denoiser Scheduler
        self.cond_encoder = cond_encoder         # Condition Encoder
        self.mlx_dino_encoder = mlx_dino_encoder
        self.mlx_control_encoder = mlx_control_encoder
        self.image_processor = image_processor   # Image Processor
        self.scale_factor = scale_factor
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.fast_decode = bool(kwargs.get("fast_decode", False))
        self.native_mlx = bool(kwargs.get("native_mlx", False))
        self.offload_mlx_dino_after_encode = bool(kwargs.get("offload_mlx_dino_after_encode", False))

    def _has_guidance_embed(self):
        model_params = getattr(self.model, "params", None)
        return bool(getattr(model_params, "guidance_embed", False))

    @staticmethod
    def _generate_voxel(pc, resolution=16):
        device, dtype = pc.device, pc.dtype
        points_norm = (pc + 1) / 2
        voxels = (points_norm * resolution).floor().long()
        voxels = torch.clamp(voxels, 0, resolution - 1)

        sampled_voxels_batch = []
        for b in range(pc.shape[0]):
            vox_b = voxels[b]
            linear_idx = vox_b[:, 0] + vox_b[:, 1] * resolution + vox_b[:, 2] * resolution * resolution
            unique_idx = torch.unique(linear_idx)
            z = unique_idx // (resolution * resolution)
            y = (unique_idx % (resolution * resolution)) // resolution
            x = unique_idx % resolution
            unique_voxels = torch.stack([x, y, z], dim=1).float()
            voxel_size = 2.0 / resolution
            sampled_voxels_batch.append(unique_voxels * voxel_size + voxel_size / 2 - 1)

        max_voxels = max(v.shape[0] for v in sampled_voxels_batch)
        padded_voxels = []
        for vox in sampled_voxels_batch:
            pad_len = max_voxels - vox.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, 3, device=device, dtype=dtype)
                vox = torch.cat([vox.to(device=device, dtype=dtype), pad], dim=0)
            else:
                vox = vox.to(device=device, dtype=dtype)
            padded_voxels.append(vox.unsqueeze(0))
        return torch.cat(padded_voxels, dim=0)

    def encode_cond(self, image, surface, pose, bbox, point, voxel, do_classifier_free_guidance):
        bsz = image.shape[0]
        if self.native_mlx and self.mlx_control_encoder is not None:
            import mlx.core as mx

            if self.mlx_dino_encoder is not None:
                image_cond = self.mlx_dino_encoder(image)
            else:
                if self.cond_encoder is None:
                    raise RuntimeError("Native MLX DINO was offloaded and no PyTorch conditioner is available.")
                image_cond = self.cond_encoder.image_encoder(
                    image,
                    dropout_mask=None,
                    mask=None,
                )['dino']['last_hidden_state']
            sampled_voxel = None
            if voxel is not None:
                if self.cond_encoder is not None:
                    sampled_voxel = self.cond_encoder.generate_voxel(voxel[..., :3])
                else:
                    sampled_voxel = self._generate_voxel(voxel[..., :3])
                voxel = sampled_voxel
            cond = self.mlx_control_encoder(
                image_cond,
                pose=pose,
                bbox=bbox,
                point=point,
                voxel=voxel,
                sampled_point=sampled_voxel,
            )
            sampled_point = cond['cond_point']
            if do_classifier_free_guidance:
                image_len = image_cond.shape[1]
                img_uncond = mx.zeros((bsz, image_len, image_cond.shape[2]), dtype=_torch_dtype_to_mlx(self.dtype))
                control_cond = cond['cond'][:, image_len:, :]
                uncond = mx.concatenate([img_uncond, control_cond], axis=1)
                cond = {'cond': mx.concatenate([cond['cond'], uncond], axis=0)}
            cond['cond'] = mx.contiguous(cond['cond'])
            mx.eval(cond['cond'])
            if self.offload_mlx_dino_after_encode and self.mlx_dino_encoder is not None:
                self.mlx_dino_encoder.release()
                self.mlx_dino_encoder = None
                gc.collect()
                mx.clear_cache()
            return cond, sampled_point

        cond = self.cond_encoder(image=image, surface=surface, pose=pose, bbox=bbox, point=point, voxel=voxel)
        sampled_point = cond['cond_point']
        if do_classifier_free_guidance:
            img_uncond = self.cond_encoder.unconditional_embedding(bsz)["dino"]["last_hidden_state"]
            uncond = torch.cat([img_uncond, cond['cond'][:, img_uncond.shape[1]:, :]], dim=1)
            un_cond = {'cond': uncond}
            cond = {'cond': torch.cat((cond['cond'], un_cond['cond']), dim=0)}
        return cond, sampled_point

    def prepare_latents(self, timestep, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_mlx_latents(self, batch_size, dtype, generator, latents=None):
        import mlx.core as mx

        shape = (batch_size, *self.vae.latent_shape)
        if latents is None:
            if generator is not None and hasattr(generator, "initial_seed"):
                mx.random.seed(int(generator.initial_seed()))
            latents = mx.random.normal(shape, dtype=_torch_dtype_to_mlx(dtype))
        elif isinstance(latents, torch.Tensor):
            latents = _torch_tensor_to_mlx(latents, dtype)
        else:
            latents = latents.astype(_torch_dtype_to_mlx(dtype))
        return latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)


    def prepare_image(self, image, mask):
        if isinstance(image, torch.Tensor):
            return image, mask

        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Image path {image} does not exist.")
            image = [image]

        image_pts, mask_pts = [], []
        for img in image:
            image_pt, mask_pt = self.image_processor(img, return_mask=True, return_view_idx=False)
            image_pts.append(image_pt)
            mask_pts.append(mask_pt)

        image_pts = torch.cat(image_pts, dim=0).to(self.device, dtype=self.dtype)
        if mask_pts[0] is not None:
            mask_pts = torch.cat(mask_pts, dim=0).to(self.device, dtype=self.dtype)
        else:
            mask_pts = None
        return image_pts, mask_pts

    def _sample_mlx_euler(
        self,
        latents,
        cond,
        guidance,
        do_classifier_free_guidance,
        guidance_scale,
        num_inference_steps,
        sampling_method,
        return_mlx=False,
    ):
        import mlx.core as mx

        if sampling_method.lower() != "euler":
            raise ValueError("The native MLX denoiser currently supports sampling_method='euler'.")

        transport = getattr(self.scheduler, "transport", None)
        if transport is None:
            raise ValueError("The native MLX denoiser requires a transport-backed scheduler.")
        t0, t1 = transport.check_interval(
            transport.train_eps,
            transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        mx_latents = _torch_tensor_to_mlx(latents, self.dtype) if isinstance(latents, torch.Tensor) else latents.astype(_torch_dtype_to_mlx(self.dtype))
        mx_contexts = {
            key: _torch_tensor_to_mlx(value, self.dtype) if isinstance(value, torch.Tensor) else value
            for key, value in cond.items()
        }
        mx_contexts = {
            key: mx.contiguous(value) if hasattr(value, "shape") else value
            for key, value in mx_contexts.items()
        }
        mx_guidance = _torch_tensor_to_mlx(guidance, self.dtype) if isinstance(guidance, torch.Tensor) else None
        mx.eval(mx_latents, *[value for value in mx_contexts.values() if hasattr(value, "shape")])
        timesteps = mx.linspace(float(t0), float(t1), num_inference_steps)

        for step_idx in tqdm(range(num_inference_steps - 1), desc="MLX denoise"):
            t_cur = mx.full((mx_latents.shape[0],), timesteps[step_idx], dtype=_torch_dtype_to_mlx(self.dtype))
            noise_pred = self.model(mx_latents, t_cur, mx_contexts, guidance_cond=mx_guidance)
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = mx.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = mx.concatenate([noise_pred, noise_pred], axis=0)
            dt = timesteps[step_idx + 1] - timesteps[step_idx]
            mx_latents = mx_latents + dt * noise_pred
            mx.eval(mx_latents)

        if return_mlx:
            return mx_latents
        return _mlx_array_to_torch(mx_latents, self.device, self.dtype)

    @torch.no_grad()
    def __call__(
        self,
        image,
        mask: torch.Tensor = None,
        surface: Union[str, List[str], torch.Tensor] = None,
        pose: Union[str, List[str], torch.Tensor] = None,
        bbox: Union[str, List[str], torch.Tensor] = None,
        point: Union[str, List[str], torch.Tensor] = None,
        voxel: Union[str, List[str], torch.Tensor] = None,
        fast_decode: bool = None,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        prev_guidance_scale: float = None,
        guidance_scale: float = 7.5,
        dino_image_size: int = None,
        generator=None,
        box_v=1.01,
        octree_depth=8,
        octree_resolution=None,
        mc_level=0.0,
        mc_mode='mc',
        num_chunks=8000,
        sigmoid=False,
        output_type: Optional[str] = "trimesh",
        sampling_method='euler',
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        '''
            Generate 3D shapes using the Hunyuan3DOmni model with SiT flow matching.
            Args:
                image: the input image.
                mask: the input mask.
                surface: the input surface.
                pose: the input pose.
                bbox: the input bbox.
                point: the input point.
                voxel: the input voxel.
                fast_decode: whether to use fast decode (flashvdm).
                prompt: the input prompt.
                num_inference_steps: the number of inference steps.
                timesteps: the timesteps.
                prev_guidance_scale: the previous guidance scale.
                guidance_scale: the guidance scale.
                dino_image_size: the dino image size.
                generator: the generator.
                box_v: the box v.
                octree_depth: the octree depth.
                octree_resolution: the octree resolution.
                mc_level: the mc level.
                mc_mode: the mc mode.
                num_chunks: the number of chunks.
                sigmoid: whether to use sigmoid.
                output_type: the output type.
        '''

        if fast_decode is None:
            fast_decode = self.fast_decode
        self.vae.fast_decode = bool(fast_decode)

        if dino_image_size is not None and hasattr(self.cond_encoder, 'setup'):
            self.cond_encoder.setup(image_size=dino_image_size)

        device, dtype = self.device, self.dtype

        do_classifier_free_guidance = guidance_scale >= 0 and not self._has_guidance_embed()

        image, mask = self.prepare_image(image, mask)
        cond, sampled_point = self.encode_cond(image, surface, pose, bbox, point, voxel, do_classifier_free_guidance)
        batch_size = image.shape[0]

        use_native_mlx_vae = self.native_mlx and hasattr(self.vae, "decode_latents")
        if use_native_mlx_vae:
            latents = self.prepare_mlx_latents(batch_size, dtype, generator)
        else:
            latents = self.prepare_latents(None, batch_size, dtype, device, generator)
        sample_fn = None if self.native_mlx else self.scheduler.sample_ode(num_steps=num_inference_steps, sampling_method=sampling_method)
        guidance = None
        if self._has_guidance_embed():
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            print("[Guidance Distilled Model] Using guidance embeddings.")

        def denoise_with_cfg(inputs, ts, contexts):
            noise_pred = self.model(inputs, ts, contexts, guidance=guidance)
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return torch.cat([noise_pred, noise_pred], dim=0)
            return noise_pred

        if do_classifier_free_guidance:
            if isinstance(latents, torch.Tensor):
                latents = torch.cat([latents, latents])
            else:
                import mlx.core as mx
                latents = mx.concatenate([latents, latents], axis=0)

        if self.native_mlx:
            latents = self._sample_mlx_euler(
                latents,
                cond,
                guidance,
                do_classifier_free_guidance,
                guidance_scale,
                num_inference_steps,
                sampling_method,
                return_mlx=use_native_mlx_vae,
            )
        else:
            latents = sample_fn(latents, denoise_with_cfg, contexts=cond)[-1]
        if do_classifier_free_guidance:
            if isinstance(latents, torch.Tensor):
                latents = latents.chunk(2, dim=0)[0]
            else:
                import mlx.core as mx
                latents = mx.split(latents, 2, axis=0)[0]

        if not output_type == "latent":
            latents = 1. / self.scale_factor * latents
            if isinstance(latents, torch.Tensor):
                print("converting latents dtype to ", next(self.vae.parameters()).dtype)
                latents = latents.to(next(self.vae.parameters()).dtype)
            else:
                print("decoding latents with native MLX ShapeVAE")

            shapes = self.vae.decode(
                latents,
                octree_depth=octree_depth,
                bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_mode=mc_mode,
                sigmoid=sigmoid,
            )
        else:
            shapes = latents

        if output_type == 'trimesh':
            shapes = export_to_trimesh(shapes)
        return {'shapes':[shapes], 'sampled_point':sampled_point, 'image': image}

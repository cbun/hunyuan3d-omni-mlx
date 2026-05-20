# -*- coding: utf-8 -*-
"""Gradio UI for Hunyuan3D-Omni-MLX inference."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import random
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import numpy as np
import torch
import trimesh

from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover
from hy3dshape.runtime import describe_backend, has_mlx, make_generator, resolve_device, resolve_dtype


DEFAULT_REPO_ID = "tencent/Hunyuan3D-Omni"
DEFAULT_OUTPUT_DIR = "gradio_outputs"
_PIPELINE_CACHE: Dict[Tuple[Any, ...], Hunyuan3DOmniSiTFlowMatchingPipeline] = {}


@dataclass(frozen=True)
class ModelSettings:
    repo_id: str
    variant: Optional[str]
    device_choice: str
    dtype_choice: str
    fast_decode: bool
    native_mlx_dino: bool
    offload_mlx_dino_after_encode: bool
    sparse_mlx_moe: bool

    @property
    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.repo_id,
            self.variant,
            self.device_choice,
            self.dtype_choice,
            self.fast_decode,
            self.native_mlx_dino,
            self.offload_mlx_dino_after_encode,
            self.sparse_mlx_moe,
        )


def default_device_choice() -> str:
    if platform.system() == "Darwin" and has_mlx():
        return "mlx"
    return "auto"


def _clean_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "run"


def _make_model_settings(
    repo_id: str,
    variant: str,
    device_choice: str,
    dtype_choice: str,
    fast_decode: bool,
    native_mlx_dino: bool,
    offload_mlx_dino_after_encode: bool,
    sparse_mlx_moe: bool,
) -> ModelSettings:
    repo_id = (repo_id or DEFAULT_REPO_ID).strip()
    if not repo_id:
        repo_id = DEFAULT_REPO_ID
    selected_variant = None if variant == "default" else variant
    return ModelSettings(
        repo_id=repo_id,
        variant=selected_variant,
        device_choice=device_choice,
        dtype_choice=dtype_choice,
        fast_decode=bool(fast_decode),
        native_mlx_dino=bool(native_mlx_dino),
        offload_mlx_dino_after_encode=bool(offload_mlx_dino_after_encode),
        sparse_mlx_moe=bool(sparse_mlx_moe),
    )


def _resolve_runtime(settings: ModelSettings) -> Tuple[torch.device, torch.dtype, Any, str]:
    try:
        resolved_device = resolve_device(settings.device_choice)
        resolved_dtype = resolve_dtype(settings.dtype_choice, resolved_device)
    except (RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    pipeline_device_arg = settings.device_choice if settings.device_choice == "mlx" else resolved_device
    if resolved_device.type != "cuda" and settings.fast_decode:
        raise gr.Error("FlashVDM is CUDA-oriented. Disable FlashVDM for CPU, MPS, or MLX.")
    backend = describe_backend(settings.device_choice if settings.device_choice == "mlx" else resolved_device, resolved_dtype)
    return resolved_device, resolved_dtype, pipeline_device_arg, backend


def _get_pipeline(settings: ModelSettings, progress: Optional[gr.Progress] = None) -> Tuple[Hunyuan3DOmniSiTFlowMatchingPipeline, str, bool]:
    _, resolved_dtype, pipeline_device_arg, backend = _resolve_runtime(settings)
    cache_key = settings.cache_key
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key], backend, False

    if progress is not None:
        progress(0.05, desc="Loading model")

    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        settings.repo_id,
        variant=settings.variant,
        device=pipeline_device_arg,
        dtype=resolved_dtype,
        fast_decode=settings.fast_decode,
        native_mlx_dino=settings.native_mlx_dino,
        offload_mlx_dino_after_encode=settings.offload_mlx_dino_after_encode,
        sparse_mlx_moe=settings.sparse_mlx_moe,
    )
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline, backend, True


def _clear_pipeline_cache() -> None:
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _ensure_file(path: Optional[str], label: str) -> str:
    if not path:
        raise gr.Error(f"{label} is required.")
    if not os.path.exists(path):
        raise gr.Error(f"{label} does not exist: {path}")
    return path


def _result_directory(output_dir: str, control_type: str) -> Path:
    root = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = root / control_type / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_ply_points(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")


def _as_numpy_points(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    try:
        import mlx.core as mx

        if isinstance(value, mx.array):
            mx.eval(value)
            return np.array(value).astype(np.float32)
    except Exception:
        pass
    return np.asarray(value, dtype=np.float32)


def _normalize_mesh(mesh: trimesh.Trimesh, scale: float = 0.9999) -> trimesh.Trimesh:
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise gr.Error("The uploaded control geometry has no vertices.")
    bounds = mesh.bounds
    center = (bounds[1] + bounds[0]) / 2
    extent = float((bounds[1] - bounds[0]).max())
    if extent <= 0:
        raise gr.Error("The uploaded control geometry has zero size.")
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / extent * 2 * scale)
    return mesh


def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise gr.Error(f"Could not load a mesh or point cloud from {path}.")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise gr.Error(f"The uploaded file has no vertices: {path}")
    return mesh


def _sample_surface_or_vertices(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    count = int(count)
    if count <= 0:
        raise gr.Error("Sample count must be positive.")
    if mesh.faces is not None and len(mesh.faces) > 0:
        np.random.seed(seed)
        return mesh.sample(count).astype(np.float32)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) >= count:
        rng = np.random.default_rng(seed)
        return vertices[rng.choice(len(vertices), size=count, replace=False)]
    repeats = int(np.ceil(count / len(vertices)))
    return np.tile(vertices, (repeats, 1))[:count]


def _tensor_from_numpy(points: np.ndarray, pipeline: Hunyuan3DOmniSiTFlowMatchingPipeline) -> torch.Tensor:
    return torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(device=pipeline.device, dtype=pipeline.dtype)


def _extract_first_mesh(result: Dict[str, Any]) -> Optional[trimesh.Trimesh]:
    shapes = result.get("shapes", [])
    if not shapes:
        return None
    mesh = shapes[0]
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0] if mesh else None
    if mesh is None:
        return None
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    return mesh


def _postprocess_and_save(
    result: Dict[str, Any],
    image_path: str,
    run_dir: Path,
    control_type: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    mesh = _extract_first_mesh(result)
    sampled_point = result.get("sampled_point")
    if isinstance(sampled_point, (list, tuple)):
        sampled_point = sampled_point[0] if sampled_point else None
    sampled_points = _as_numpy_points(sampled_point).reshape(-1, 3)

    base_name = _clean_name(Path(image_path).stem)
    stem = f"{control_type}_{base_name}"
    ply_path = run_dir / f"{stem}_sampled_points.ply"
    image_copy_path = run_dir / f"{stem}{Path(image_path).suffix or '.png'}"
    _save_ply_points(ply_path, sampled_points)
    shutil.copy(image_path, image_copy_path)

    metadata: Dict[str, Any] = {
        "output_dir": str(run_dir),
        "sampled_points": int(len(sampled_points)),
        "sampled_ply": str(ply_path),
        "input_image": str(image_copy_path),
    }

    if mesh is None or len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
        metadata["mesh_status"] = "no_valid_surface"
        return None, str(ply_path), str(image_copy_path), metadata

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    glb_path = run_dir / f"{stem}.glb"
    mesh.export(glb_path)
    metadata.update(
        {
            "mesh_status": "ok",
            "glb": str(glb_path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds": np.asarray(mesh.bounds, dtype=float).round(6).tolist(),
        }
    )
    return str(glb_path), str(ply_path), str(image_copy_path), metadata


def _run_generation(
    control_type: str,
    image_path: str,
    control_kwargs: Dict[str, Any],
    repo_id: str,
    variant: str,
    device_choice: str,
    dtype_choice: str,
    fast_decode: bool,
    native_mlx_dino: bool,
    offload_mlx_dino_after_encode: bool,
    sparse_mlx_moe: bool,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    randomize_seed: bool,
    octree_resolution: int,
    num_chunks: int,
    mc_mode: str,
    box_v: float,
    output_dir: str,
    progress: gr.Progress,
) -> Tuple[Optional[str], Optional[str], Optional[str], str, Dict[str, Any]]:
    start = time.perf_counter()
    image_path = _ensure_file(image_path, "Input image")
    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)

    settings = _make_model_settings(
        repo_id,
        variant,
        device_choice,
        dtype_choice,
        fast_decode,
        native_mlx_dino,
        offload_mlx_dino_after_encode,
        sparse_mlx_moe,
    )

    pipeline, backend, loaded = _get_pipeline(settings, progress)
    progress(0.20, desc="Preparing controls")
    prepared_controls = control_kwargs["prepare"](pipeline, seed)
    run_dir = _result_directory(output_dir, control_type)

    progress(0.30, desc="Generating mesh")
    generator = make_generator(pipeline.device, seed)
    result = pipeline(
        image=image_path,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=int(num_inference_steps),
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        mc_mode=mc_mode,
        fast_decode=bool(fast_decode),
        box_v=float(box_v),
        mc_level=0,
        **prepared_controls,
    )

    progress(0.90, desc="Saving outputs")
    glb_path, ply_path, image_copy_path, metadata = _postprocess_and_save(result, image_path, run_dir, control_type)
    elapsed = time.perf_counter() - start
    metadata.update(
        {
            "control_type": control_type,
            "backend": backend,
            "model_loaded": loaded,
            "repo_id": settings.repo_id,
            "variant": settings.variant or "default",
            "device": settings.device_choice,
            "dtype": settings.dtype_choice,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "octree_resolution": int(octree_resolution),
            "num_chunks": int(num_chunks),
            "mc_mode": mc_mode,
            "elapsed_seconds": round(elapsed, 2),
        }
    )

    if glb_path is None:
        status = (
            "Generation completed, but marching cubes did not produce a valid surface. "
            "The sampled control points were saved; try more denoising steps or a higher octree resolution."
        )
    else:
        status = (
            f"Generated {metadata['vertices']:,} vertices and {metadata['faces']:,} faces "
            f"in {metadata['elapsed_seconds']}s on {backend}."
        )
    return glb_path, glb_path, ply_path, status, metadata


def _bbox_prepare(width: float, height: float, depth: float):
    def prepare(pipeline: Hunyuan3DOmniSiTFlowMatchingPipeline, seed: int) -> Dict[str, Any]:
        bbox = torch.tensor([width, height, depth], dtype=torch.float32).reshape(1, 1, 3)
        bbox = bbox.to(device=pipeline.device, dtype=pipeline.dtype)
        return {"bbox": bbox}

    return prepare


def _pose_prepare(pose_file: str):
    pose_file = _ensure_file(pose_file, "Pose file")

    def prepare(pipeline: Hunyuan3DOmniSiTFlowMatchingPipeline, seed: int) -> Dict[str, Any]:
        pose = np.loadtxt(pose_file).astype(np.float32)
        if pose.ndim == 1:
            pose = pose.reshape(1, -1)
        if pose.shape[-1] != 6:
            raise gr.Error("Pose files must have six columns per row: x1 y1 z1 x2 y2 z2.")
        tensor = torch.from_numpy(pose).unsqueeze(0).to(device=pipeline.device, dtype=pipeline.dtype)
        return {"pose": tensor}

    return prepare


def _point_prepare(control_file: str):
    control_file = _ensure_file(control_file, "Point/mesh control file")

    def prepare(pipeline: Hunyuan3DOmniSiTFlowMatchingPipeline, seed: int) -> Dict[str, Any]:
        mesh = _normalize_mesh(_load_mesh(control_file), scale=0.98)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        return {"point": _tensor_from_numpy(points, pipeline)}

    return prepare


def _voxel_prepare(control_file: str, sample_count: int):
    control_file = _ensure_file(control_file, "Voxel/mesh control file")

    def prepare(pipeline: Hunyuan3DOmniSiTFlowMatchingPipeline, seed: int) -> Dict[str, Any]:
        mesh = _load_mesh(control_file)
        rotation = trimesh.transformations.rotation_matrix(angle=np.radians(-90), direction=[1, 0, 0])
        mesh.apply_transform(rotation)
        mesh = _normalize_mesh(mesh)
        points = _sample_surface_or_vertices(mesh, int(sample_count), seed)
        return {"voxel": _tensor_from_numpy(points, pipeline)}

    return prepare


def _model_status(
    repo_id: str,
    variant: str,
    device_choice: str,
    dtype_choice: str,
    fast_decode: bool,
    native_mlx_dino: bool,
    offload_mlx_dino_after_encode: bool,
    sparse_mlx_moe: bool,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, Dict[str, Any]]:
    settings = _make_model_settings(
        repo_id,
        variant,
        device_choice,
        dtype_choice,
        fast_decode,
        native_mlx_dino,
        offload_mlx_dino_after_encode,
        sparse_mlx_moe,
    )
    pipeline, backend, loaded = _get_pipeline(settings, progress)
    metadata = {
        "backend": backend,
        "loaded_now": loaded,
        "cached_models": len(_PIPELINE_CACHE),
        "device": str(pipeline.device),
        "dtype": str(pipeline.dtype),
        "native_mlx": bool(getattr(pipeline, "native_mlx", False)),
    }
    verb = "Loaded" if loaded else "Using cached"
    return f"{verb} model: {settings.repo_id} on {backend}.", metadata


def _clear_model_status() -> Tuple[str, Dict[str, Any]]:
    _clear_pipeline_cache()
    return "Model cache cleared.", {"cached_models": 0}


def _common_inputs():
    repo_id = gr.Textbox(label="Model", value=DEFAULT_REPO_ID)
    with gr.Row():
        variant = gr.Dropdown(label="Weights", choices=["default", "ema"], value="default")
        device_choice = gr.Dropdown(
            label="Device",
            choices=["mlx", "mps", "cuda", "cpu", "auto"],
            value=default_device_choice(),
        )
        dtype_choice = gr.Dropdown(label="Dtype", choices=["auto", "fp16", "fp32", "bf16"], value="fp16")

    with gr.Row():
        fast_decode = gr.Checkbox(label="FlashVDM", value=False)
        native_mlx_dino = gr.Checkbox(label="Native MLX DINO", value=False)
        offload_mlx_dino = gr.Checkbox(label="Offload MLX DINO after encode", value=False)
        sparse_mlx_moe = gr.Checkbox(label="Sparse MLX MoE", value=False)

    with gr.Row():
        num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
        guidance_scale = gr.Slider(0.0, 12.0, value=4.5, step=0.1, label="Guidance")
        seed = gr.Number(value=1234, precision=0, label="Seed")
        randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

    with gr.Row():
        octree_resolution = gr.Dropdown(label="Octree resolution", choices=[64, 128, 256, 384, 512], value=512)
        num_chunks = gr.Number(value=8000, precision=0, label="Decode chunk size")
        mc_mode = gr.Dropdown(label="Mesh extraction", choices=["mc", "dmc"], value="mc")
        box_v = gr.Slider(0.8, 1.5, value=1.01, step=0.01, label="Bounds")

    with gr.Row():
        voxel_sample_count = gr.Number(value=81920, precision=0, label="Voxel samples")
        output_dir = gr.Textbox(label="Output folder", value=DEFAULT_OUTPUT_DIR)

    model_inputs = [
        repo_id,
        variant,
        device_choice,
        dtype_choice,
        fast_decode,
        native_mlx_dino,
        offload_mlx_dino,
        sparse_mlx_moe,
    ]
    generation_inputs = [
        num_inference_steps,
        guidance_scale,
        seed,
        randomize_seed,
        octree_resolution,
        num_chunks,
        mc_mode,
        box_v,
        output_dir,
    ]
    return model_inputs, generation_inputs, voxel_sample_count


def _bind_generate(
    button: gr.Button,
    fn,
    inputs,
    outputs,
) -> None:
    button.click(fn=fn, inputs=inputs, outputs=outputs)


def build_demo() -> gr.Blocks:
    css = """
    #status { min-height: 48px; }
    .compact-note { color: var(--body-text-color-subdued); font-size: 0.92rem; }
    """
    with gr.Blocks(title="Hunyuan3D-Omni-MLX", css=css) as demo:
        gr.Markdown(
            "# Hunyuan3D-Omni-MLX\n"
            "Apple Silicon optimized controllable 3D generation with bbox, pose, point, and voxel controls."
        )

        with gr.Accordion("Model and generation settings", open=True):
            model_inputs, generation_inputs, voxel_sample_count = _common_inputs()
            with gr.Row():
                load_button = gr.Button("Load model", variant="secondary")
                clear_button = gr.Button("Clear model cache", variant="secondary")

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Bounding box"):
                        bbox_image = gr.Image(label="Image", type="filepath", sources=["upload"])
                        with gr.Row():
                            bbox_width = gr.Slider(0.05, 1.5, value=0.8, step=0.01, label="Width")
                            bbox_height = gr.Slider(0.05, 1.5, value=0.64, step=0.01, label="Height")
                            bbox_depth = gr.Slider(0.05, 1.5, value=1.0, step=0.01, label="Depth")
                        gr.Examples(
                            examples=[["demos/bbox/furniture0.png", 0.8, 0.64, 1.0]],
                            inputs=[bbox_image, bbox_width, bbox_height, bbox_depth],
                            cache_examples=False,
                        )
                        bbox_button = gr.Button("Generate from bbox", variant="primary")

                    with gr.Tab("Pose"):
                        pose_image = gr.Image(label="Image", type="filepath", sources=["upload"])
                        pose_file = gr.File(label="Pose TXT", file_count="single", file_types=[".txt"], type="filepath")
                        gr.Examples(
                            examples=[["demos/pose/263.png", "demos/pose/a_pose_bone.txt"]],
                            inputs=[pose_image, pose_file],
                            cache_examples=False,
                        )
                        pose_button = gr.Button("Generate from pose", variant="primary")

                    with gr.Tab("Point cloud"):
                        point_image = gr.Image(label="Image", type="filepath", sources=["upload"])
                        point_file = gr.File(
                            label="Point or mesh file",
                            file_count="single",
                            file_types=[".ply", ".obj", ".glb", ".stl"],
                            type="filepath",
                        )
                        gr.Examples(
                            examples=[["demos/point/imgs/008.png", "demos/point/plys/008.ply"]],
                            inputs=[point_image, point_file],
                            cache_examples=False,
                        )
                        point_button = gr.Button("Generate from points", variant="primary")

                    with gr.Tab("Voxel"):
                        voxel_image = gr.Image(label="Image", type="filepath", sources=["upload"])
                        voxel_file = gr.File(
                            label="Voxel or mesh file",
                            file_count="single",
                            file_types=[".ply", ".obj", ".glb", ".stl"],
                            type="filepath",
                        )
                        gr.Examples(
                            examples=[
                                [
                                    "demos/voxel/imgs/1c1ff58afbf4455ca80228d280f86aef.png",
                                    "demos/voxel/plys/1c1ff58afbf4455ca80228d280f86aef.ply",
                                ]
                            ],
                            inputs=[voxel_image, voxel_file],
                            cache_examples=False,
                        )
                        voxel_button = gr.Button("Generate from voxel", variant="primary")

            with gr.Column(scale=6):
                model_viewer = gr.Model3D(label="Generated GLB", height=420, clear_color=[0.04, 0.04, 0.04, 1.0])
                with gr.Row():
                    glb_file = gr.File(label="GLB")
                    ply_file = gr.File(label="Sampled PLY")
                status = gr.Markdown(elem_id="status")
                metadata = gr.JSON(label="Run metadata")

        outputs = [model_viewer, glb_file, ply_file, status, metadata]
        status_outputs = [status, metadata]

        load_button.click(fn=_model_status, inputs=model_inputs, outputs=status_outputs)
        clear_button.click(fn=_clear_model_status, inputs=None, outputs=status_outputs)

        shared = model_inputs + generation_inputs

        def generate_bbox(
            image,
            width,
            height,
            depth,
            *args,
            progress=gr.Progress(track_tqdm=True),
        ):
            return _run_generation(
                "bbox",
                image,
                {"prepare": _bbox_prepare(width, height, depth)},
                *args,
                progress=progress,
            )

        def generate_pose(image, control_file, *args, progress=gr.Progress(track_tqdm=True)):
            return _run_generation(
                "pose",
                image,
                {"prepare": _pose_prepare(control_file)},
                *args,
                progress=progress,
            )

        def generate_point(image, control_file, *args, progress=gr.Progress(track_tqdm=True)):
            return _run_generation(
                "point",
                image,
                {"prepare": _point_prepare(control_file)},
                *args,
                progress=progress,
            )

        def generate_voxel(image, control_file, sample_count, *args, progress=gr.Progress(track_tqdm=True)):
            return _run_generation(
                "voxel",
                image,
                {"prepare": _voxel_prepare(control_file, int(sample_count))},
                *args,
                progress=progress,
            )

        _bind_generate(
            bbox_button,
            generate_bbox,
            [bbox_image, bbox_width, bbox_height, bbox_depth] + shared,
            outputs,
        )
        _bind_generate(pose_button, generate_pose, [pose_image, pose_file] + shared, outputs)
        _bind_generate(point_button, generate_point, [point_image, point_file] + shared, outputs)
        _bind_generate(
            voxel_button,
            generate_voxel,
            [voxel_image, voxel_file, voxel_sample_count] + shared,
            outputs,
        )

    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Hunyuan3D-Omni-MLX Gradio UI.")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
    )

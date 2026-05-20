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

import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Load libgcc_s on Linux to prevent runtime issues. This library is not present
# on macOS, so keep it optional for Apple Silicon.
if platform.system() == "Linux":
    import ctypes
    try:
        ctypes.CDLL('libgcc_s.so.1')
    except OSError:
        pass

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')

import io
import json
import random
import argparse
import torch
import trimesh
import glob
import shutil
import numpy as np
from PIL import Image
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.preprocessors import ImageProcessorV2
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover
from hy3dshape.runtime import describe_backend, make_generator, resolve_device, resolve_dtype


def save_ply_points(filename: str, points: np.ndarray) -> None:
    """
    Save 3D points to a PLY format file.
    
    This function exports a point cloud to the PLY (Polygon File Format) which
    can be viewed in 3D visualization software like MeshLab or Blender.
    
    Args:
        filename (str): Output PLY file path
        points (np.ndarray): Array of 3D points with shape [N, 3]
        
    Returns:
        None
    """
    with open(filename, 'w') as f:
        # Write PLY header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        
        # Write point coordinates
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 0.9999) -> trimesh.Trimesh:
    """
    Normalize a 3D mesh to fit within a centered cube.
    
    This function centers the mesh at the origin and scales it to fit within
    a cube of the specified scale range [-scale, scale]. This normalization
    is essential for consistent model training and inference.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh to normalize
        scale (float): Target scale range. Mesh will fit in [-scale, scale]. Defaults to 0.9999.
        
    Returns:
        trimesh.Trimesh: Normalized mesh centered at origin
    """
    # Calculate bounding box
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2  # Center point
    scale_ = (bbox[1] - bbox[0]).max()  # Maximum dimension
    
    # Apply centering and scaling transformations
    mesh.apply_translation(-center)  # Move to origin
    mesh.apply_scale(1 / scale_ * 2 * scale)  # Scale to target range
    return mesh

def postprocess(mesh, file_name, save_dir, sampled_point, image_file):
    if mesh is None or len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
        print(f"Warning: no mesh surface generated for {file_name}; saving point cloud and input image only.")
        save_ply_points(os.path.join(save_dir, '%s.ply' % file_name), sampled_point.detach().cpu().numpy())
        shutil.copy(image_file, os.path.join(save_dir, '%s.png' % file_name))
        return False
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh.export(os.path.join(save_dir, '%s.glb' % (file_name)))
    save_ply_points(os.path.join(save_dir, '%s.ply' % file_name), sampled_point.detach().cpu().numpy())
    shutil.copy(image_file, os.path.join(save_dir, '%s.png' % file_name))
    return True

def infer_bbox(pipeline, data_json: str, save_dir: str, seed: int, inference_kwargs: dict, max_items: int = None) -> None:
    """
    Perform 3D generation with bounding box control.
    
    This function generates 3D models conditioned on input images and 3D bounding boxes.
    The bounding box provides spatial constraints for the generated geometry.
    
    Args:
        gpu_id (int): GPU device ID to use for inference
        num_gpu (int): Total number of GPUs (for potential distributed inference)
        data_json (str): Path to JSON file containing image paths and bounding box data
        save_dir (str): Directory to save generated 3D models and outputs
        
    Returns:
        None
        
    Expected JSON format:
        {
            "image": ["path/to/image1.png", "path/to/image2.png", ...],
            "bbox": [[x_min, y_min, z_min, x_max, y_max, z_max], ...]
        }
        
    Outputs:
        - {filename}_{bbox_coords}.glb: Generated 3D mesh in GLB format
        - {filename}_{bbox_coords}.ply: Point cloud representation
        - {filename}_{bbox_coords}.png: Copy of input image
    """
    # Load input data from JSON file
    data = json.load(open(data_json))
    image_files = data['image']
    bboxs = data['bbox']
    
    total_files = len(image_files) if max_items is None else min(len(image_files), max_items)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing {total_files} images with bounding box control...")

    # Process each image-bbox pair
    for i in range(total_files):
        image_file = image_files[i]
        bbox = bboxs[i]
        print(f"Processing: {image_file}")

        # Validate input file exists
        if not os.path.exists(image_file):
            print(f"Warning: Image file {image_file} does not exist, skipping...")
            continue
            
        # Prepare bounding box tensor [1, 1, 6] format
        bbox = torch.FloatTensor(bbox).unsqueeze(0).unsqueeze(0).to(pipeline.device).to(pipeline.dtype)
        print(f"Bounding box shape: {bbox.shape}")

        # Run inference with bounding box conditioning
        result = pipeline(
            image=image_file,
            bbox=bbox,
            mc_level=0,                  # Marching cubes iso-level
            guidance_scale=4.5,          # Classifier-free guidance strength
            generator=make_generator(pipeline.device, seed),
            **inference_kwargs,
        )
        
        # Extract results
        mesh = result['shapes'][0][0]  # Generated 3D mesh
        sampled_point = result['sampled_point'][0]  # Sampled point cloud
        
        print(f"Generated mesh: {type(mesh)}")
        print(f"Mesh info: {mesh}")
        
        # Generate output filename with bbox coordinates
        base_name = image_file.split("/")[-1].split('.')[0]
        bbox_coords = f"{bbox[0][0][0].item()}_{bbox[0][0][1].item()}_{bbox[0][0][2].item()}"
        file_name = f"{base_name}_{bbox_coords}"
        
        # Save outputs
        postprocess(mesh, file_name, save_dir, sampled_point, image_file)


def infer_pose(pipeline, images: list, pose_dict: dict, save_dir: str, seed: int, inference_kwargs: dict, max_items: int = None) -> None:
    """
    Perform 3D generation with skeletal pose control.
    
    This function generates 3D human models conditioned on input images and skeletal pose
    information. The pose control allows for generating characters in specific poses.
    
    Args:
        gpu_id (int): GPU device ID to use for inference
        num_gpu (int): Total number of GPUs (for potential distributed inference)
        images (list): List of input image file paths
        pose_dict (dict): Dictionary mapping pose names to bone point JSON file paths
        save_dir (str): Directory to save generated 3D models and outputs
        
    Returns:
        None
        
    Expected pose_dict format:
        {
            "pose_name1": "path/to/bone_points_v2.json",
            "pose_name2": "path/to/bone_points_v2.json",
            ...
        }
        
    Outputs:
        - {filename}_{pose_name}.glb: Generated 3D mesh in GLB format
        - {filename}_{pose_name}.ply: Point cloud representation
        - {filename}_{pose_name}.png: Copy of input image
    """
    
    os.makedirs(save_dir, exist_ok=True)
    total_files = len(images) if max_items is None else min(len(images), max_items)
    
    for i in range(total_files):
        image_file = images[i]
        print(image_file)
        if not os.path.exists(image_file):
            print(f"文件路径{image_file}不存在。")
            continue
        for pose_key in pose_dict.keys():
            bone_path = pose_dict[pose_key]

            bone_points = torch.from_numpy(np.loadtxt(bone_path).astype(np.float32)).to(device=pipeline.device, dtype=pipeline.dtype).unsqueeze(0)
            print(f"pose: {bone_points.shape}")

            result = pipeline(
                image=image_file,
                pose=bone_points,
                mc_level=0,
                guidance_scale=4.5,
                generator=make_generator(pipeline.device, seed),
                **inference_kwargs,
            )
            mesh = result['shapes'][0][0]
            sampled_point = result['sampled_point'][0] 
            file_name = image_file.split("/")[-1].split('.')[0] + "_" + pose_key
            postprocess(mesh, file_name, save_dir, sampled_point, image_file)


def infer_point(pipeline, data_json: str, save_dir: str, seed: int, inference_kwargs: dict, max_items: int = None) -> None:
    """
    Perform 3D generation with point cloud control.
    
    This function generates 3D models conditioned on input images and reference point clouds.
    The point cloud provides geometric guidance for the generated 3D structure.
    
    Args:
        gpu_id (int): GPU device ID to use for inference
        num_gpu (int): Total number of GPUs for distributed processing
        data_json (str): Path to JSON file containing image paths and point cloud data
        save_dir (str): Directory to save generated 3D models and outputs
        
    Returns:
        None
        
    Expected JSON format:
        {
            "image": ["path/to/image1.png", "path/to/image2.png", ...],
            "point": ["path/to/pointcloud1.ply", "path/to/pointcloud2.obj", ...]
        }
        
    Outputs:
        - {uid}/{filename}.glb: Generated 3D mesh in GLB format
        - {uid}/{filename}.ply: Point cloud representation
        - {uid}/{filename}.png: Copy of input image
    """
    data = json.load(open(data_json))
    image_files = data['image']
    mesh_files = data['point']

    # Split data across multiple GPUs
    total_files = len(image_files) if max_items is None else min(len(image_files), max_items)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(total_files):
        data_uid = image_files[i].split("/")[-2]
        save_dir_uid = os.path.join(save_dir, data_uid)
        os.makedirs(save_dir_uid, exist_ok=True)
        mesh_file = mesh_files[i]
        image_file = image_files[i]
        print(image_file)
        
        if not os.path.exists(mesh_file):
            print(f"文件路径{mesh_file}不存在。")
            continue
        if not os.path.exists(image_file):
            print(f"文件路径{image_file}不存在。")
            continue


        mesh = trimesh.load(mesh_file)
        mesh = normalize_mesh(mesh, scale=0.98)
        surface = mesh.vertices
        # surface[:, 2] = surface[:, 2] + 0.3
        surface = torch.FloatTensor(surface).unsqueeze(0)
        surface = surface.to(pipeline.device).to(pipeline.dtype)

        result = pipeline(
            image=image_file,
            point=surface,
            mc_level=0,
            guidance_scale=4.5,
            generator=make_generator(pipeline.device, seed),
            **inference_kwargs,
        )
        mesh = result['shapes'][0][0]#[0]
        sampled_point = result['sampled_point'][0]#[0]

        file_name = image_file.split("/")[-1].split('.')[0]
        postprocess(mesh, file_name, save_dir, sampled_point, image_file)

def infer_voxel(pipeline, data_json: str, save_dir: str, seed: int, inference_kwargs: dict, max_items: int = None) -> None:
    """
    Perform 3D generation with voxel control.
    
    This function generates 3D models conditioned on input images and voxel representations.
    The voxel input provides volumetric guidance for the generated 3D structure.
    
    Args:
        gpu_id (int): GPU device ID to use for inference
        num_gpu (int): Total number of GPUs for distributed processing
        data_json (str): Path to JSON file containing image paths and voxel data
        save_dir (str): Directory to save generated 3D models and outputs
        
    Returns:
        None
        
    Expected JSON format:
        {
            "image": ["path/to/image1.png", "path/to/image2.png", ...],
            "voxel": ["path/to/voxel1.obj", "path/to/voxel2.ply", ...]
        }
        
    Outputs:
        - {uid}/{filename}.glb: Generated 3D mesh in GLB format (with post-processing)
        - {uid}/{filename}.ply: Point cloud representation
        - {uid}/{filename}.png: Copy of input image
        
    Note:
        This function includes post-processing with FloaterRemover and DegenerateFaceRemover
        for cleaner output meshes.
    """
    data = json.load(open(data_json))
    image_files = data['image']
    mesh_files= data['voxel']

    # Split data across multiple GPUs
    total_files = len(image_files) if max_items is None else min(len(image_files), max_items)

    os.makedirs(save_dir, exist_ok=True)

    for i in range(total_files):
        data_uid = image_files[i].split("/")[-2]
        save_dir_uid = os.path.join(save_dir, data_uid)
        os.makedirs(save_dir_uid, exist_ok=True)
        mesh_file = mesh_files[i]
        image_file = image_files[i]
        print(image_file)
        
        if not os.path.exists(mesh_file):
            print(f"文件路径{mesh_file}不存在。")
            continue
        if not os.path.exists(image_file):
            print(f"文件路径{image_file}不存在。")
            continue

        mesh = trimesh.load(mesh_file)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0])
        mesh.apply_transform(rotation_matrix)
        mesh = normalize_mesh(mesh)
        surface = mesh.sample(81920)
        surface = torch.FloatTensor(surface).unsqueeze(0)
        surface = surface.to(pipeline.device).to(pipeline.dtype)

        result = pipeline(
            image=image_file,
            voxel=surface,
            mc_level=0,
            guidance_scale=4.5,
            generator=make_generator(pipeline.device, seed),
            **inference_kwargs,
        )
        mesh = result['shapes'][0][0]#[0]
        # image = result['images'][0]
        sampled_point = result['sampled_point'][0]#[0]
        file_name = image_file.split("/")[-1].split('.')[0]
        # post-process
        postprocess(mesh, file_name, save_dir, sampled_point, image_file)


def get_args():
    parser = argparse.ArgumentParser(
        description='Hunyuan3D-Omni Multi-Modal Inference Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default="./omni_inference_results", help='Output directory')
    parser.add_argument('--control_type', type=str, required=True, choices=["voxel", "point", 'pose', 'bbox'])

    parser.add_argument('--repo_id', type=str, default="tencent/Hunyuan3D-Omni", help='ModelID on HuggingFace')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model for inference')
    parser.add_argument('--flashvdm', action='store_true', help='Use FlashVDM for faster decoding')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'mlx', 'cpu'],
                        help='Runtime device. Use mlx on Apple Silicon for native MLX denoising and ShapeVAE decoding with MPS conditioning.')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'],
                        help='Model dtype. auto uses fp16 on CUDA/MPS and fp32 on CPU.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducible sampling')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--octree_resolution', type=int, default=512, help='3D octree extraction resolution')
    parser.add_argument('--num_chunks', type=int, default=8000, help='Chunk size for geometry extraction')
    parser.add_argument('--mc_mode', type=str, default='mc', choices=['mc', 'dmc'],
                        help='Geometry extraction mode. Use mc on Apple Silicon; dmc requires optional diso support.')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Optional smoke-test limit for the number of demo inputs to process')
    parser.add_argument('--native_mlx_dino', action='store_true',
                        help='Experimental: use native MLX DINOv2 conditioning with --device mlx.')
    parser.add_argument('--offload_mlx_dino_after_encode', action='store_true',
                        help='Experimental one-shot mode: release native MLX DINO weights after conditioning to reduce MLX memory pressure.')
    parser.add_argument('--sparse_mlx_moe', action='store_true',
                        help='Experimental: dispatch only selected MLX MoE experts instead of evaluating all experts.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Configure command line arguments
    args = get_args()

    # Demonstration of various-condition 3D generation capabilities
    print("=" * 80)
    print("HUNYUAN3D-OMNI VARIOUS-CONDITION INFERENCE DEMO")
    print("=" * 80)

    # initial
    print(f"From Pretrained: {args.repo_id}")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        device = torch.device(f"cuda:{args.gpu_id}")
    if device.type != "cuda" and args.flashvdm:
        print("Warning: --flashvdm is a CUDA-oriented fast path; falling back to standard decoding on this backend.")
        args.flashvdm = False
    print(f"Runtime backend: {describe_backend(args.device if args.device == 'mlx' else device, dtype)}")

    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        args.repo_id,
        variant='ema' if args.use_ema else None,
        device=args.device if args.device == 'mlx' else device,
        dtype=dtype,
        fast_decode=args.flashvdm,
        native_mlx_dino=args.native_mlx_dino,
        offload_mlx_dino_after_encode=args.offload_mlx_dino_after_encode,
        sparse_mlx_moe=args.sparse_mlx_moe,
    )
    inference_kwargs = {
        "num_inference_steps": args.num_inference_steps,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "mc_mode": args.mc_mode,
        "fast_decode": args.flashvdm,
    }

    # 1. Bounding Box Control Inference
    if args.control_type == "bbox":
        print("\n" + "=" * 80)
        print("1. Running Bounding Box Control Inference...")
        bbox_data_path = "./demos/bbox/data.json"
        bbox_output_dir = os.path.join(args.save_dir, "3domni_bbox")
        infer_bbox(pipeline, bbox_data_path, bbox_output_dir, args.seed, inference_kwargs, args.max_items)
        print("Finished Bounding Box Control Inference")

    # 2. Pose Control Inference
    if args.control_type == "pose":
        print("\n" + "=" * 80)
        print("2. Running Pose Control Inference...")
        pose_configs = {
            "a_pose": "./demos/pose/a_pose_bone.txt",
            "handup_pose": "./demos/pose/handup_pose_bone.txt",
            "sky_pose": "./demos/pose/sky_pose_bone.txt",
        }
        pose_images = glob.glob("./demos/pose/*.png")
        pose_output_dir = os.path.join(args.save_dir, "3domni_pose")
        infer_pose(pipeline, pose_images, pose_configs, pose_output_dir, args.seed, inference_kwargs, args.max_items)
        print("Finished Pose Control Inference")

    # 3. Point Cloud Control Inference
    if args.control_type == "point":
        print("\n" + "=" * 80)
        print("3. Running Point Cloud Control Inference...")
        point_data_path = "./demos/point/data.json"
        point_output_dir = os.path.join(args.save_dir, "3domni_point")
        infer_point(pipeline,  point_data_path, point_output_dir, args.seed, inference_kwargs, args.max_items)
        print("Finished Point Cloud Control Inference")

    # 4. Voxel Control Inference
    if args.control_type == "voxel":
        print("\n" + "=" * 80)
        print("4. Running Voxel Control Inference...")
        voxel_data_path = "./demos/voxel/data.json"
        voxel_output_dir = os.path.join(args.save_dir, "3domni_voxel")
        infer_voxel(pipeline, voxel_data_path, voxel_output_dir, args.seed, inference_kwargs, args.max_items)
        print("Finished Voxel Control Inference")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)

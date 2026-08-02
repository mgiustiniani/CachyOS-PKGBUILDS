#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline


def parse_args():
    p = argparse.ArgumentParser(description='TRELLIS.2 ROCm image-to-3D runner')
    p.add_argument('image', help='Input image')
    p.add_argument('--out', default='/outputs/trellis2_output.glb')
    p.add_argument('--model', default=os.environ.get('TRELLIS2_MODEL_ID', 'microsoft/TRELLIS.2-4B'))
    p.add_argument('--pipeline-type', default='512', choices=['512', '1024', '1024_cascade', '1536_cascade'])
    p.add_argument('--steps', type=int, default=12, help='Default steps for all samplers')
    p.add_argument('--sparse-steps', type=int, default=None)
    p.add_argument('--shape-steps', type=int, default=None)
    p.add_argument('--texture-steps', type=int, default=None)
    p.add_argument('--preprocess', action='store_true', help='Run TRELLIS.2 preprocessing/background removal')
    p.add_argument('--low-vram', action='store_true')
    p.add_argument('--decimation-target', type=int, default=200000)
    p.add_argument('--texture-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda', torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print('device', torch.cuda.get_device_name(0), flush=True)
    print('model', args.model, flush=True)
    print('DINO_MODEL_PATH', os.environ.get('DINO_MODEL_PATH'), flush=True)
    print('HF_HOME', os.environ.get('HF_HOME'), flush=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.low_vram = args.low_vram
    pipeline.cuda()

    img = Image.open(args.image).convert('RGB')
    sparse_steps = args.sparse_steps or args.steps
    shape_steps = args.shape_steps or args.steps
    texture_steps = args.texture_steps or args.steps

    t1 = time.time()
    meshes = pipeline.run(
        img,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        preprocess_image=args.preprocess,
        sparse_structure_sampler_params={'steps': sparse_steps},
        shape_slat_sampler_params={'steps': shape_steps},
        tex_slat_sampler_params={'steps': texture_steps},
        max_num_tokens=49152,
    )
    mesh = meshes[0]
    torch.cuda.synchronize()
    print('mesh generated in', round(time.time() - t1, 2), 's', flush=True)
    print('vertices', tuple(mesh.vertices.shape), 'faces', tuple(mesh.faces.shape), flush=True)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=512 if args.pipeline_type == '512' else 1024,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    glb.export(str(out), extension_webp=True)
    print('saved', out, 'bytes', out.stat().st_size, 'total_s', round(time.time() - t0, 2), flush=True)


if __name__ == '__main__':
    main()

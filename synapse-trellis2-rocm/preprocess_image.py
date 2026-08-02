#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline


def main():
    ap = argparse.ArgumentParser(description='TRELLIS.2 preprocessing/background-removal helper')
    ap.add_argument('image')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=os.environ.get('TRELLIS2_MODEL_ID', 'microsoft/TRELLIS.2-4B'))
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('REMBG_MODEL_NAME', os.environ.get('REMBG_MODEL_NAME'))
    print('REMBG_FORCE_CPU', os.environ.get('REMBG_FORCE_CPU'))
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    img = Image.open(args.image).convert('RGB')
    out = pipeline.preprocess_image(img)
    out.save(args.out)
    print('saved', args.out, out.size, out.mode, 'elapsed_s', round(time.time() - t0, 2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Apply the minimal TRELLIS.2 runtime patches validated on Strix Halo.

The script is intentionally idempotent and uses textual edits rather than git patch
context so it can run after upstream checkout in a Docker build or manual install.
"""
from pathlib import Path

import os

ROOT = Path(os.environ.get('TRELLIS2_ROOT', '/opt/TRELLIS.2'))


def patch_image_feature_extractor():
    p = ROOT / 'trellis2/modules/image_feature_extractor.py'
    s = p.read_text()
    if 'import os\n' not in s.split('\n')[:8]:
        s = s.replace('from typing import *\n', 'from typing import *\nimport os\n')
    old = '''    def __init__(self, model_name: str, image_size=512):\n        self.model_name = model_name\n        self.model = DINOv3ViTModel.from_pretrained(model_name)\n'''
    new = '''    def __init__(self, model_name: str, image_size=512):\n        model_name = os.environ.get("DINO_MODEL_PATH", model_name)\n        self.model_name = model_name\n        self.model = DINOv3ViTModel.from_pretrained(model_name)\n'''
    if old in s:
        s = s.replace(old, new)
    elif 'DINO_MODEL_PATH' not in s:
        raise RuntimeError(f'Could not patch {p}')
    p.write_text(s)


def patch_birefnet():
    p = ROOT / 'trellis2/pipelines/rembg/BiRefNet.py'
    s = p.read_text()
    if 'import os\n' not in s.split('\n')[:8]:
        s = s.replace('from typing import *\n', 'from typing import *\nimport os\n')
    init_sig = '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
    if 'REMBG_MODEL_NAME' not in s:
        if init_sig not in s:
            raise RuntimeError(f'Could not patch BiRefNet __init__ in {p}')
        s = s.replace(
            init_sig,
            init_sig + '        model_name = os.environ.get("REMBG_MODEL_PATH", os.environ.get("REMBG_MODEL_NAME", model_name))\n        self.model_name = model_name\n',
            1,
        )

    old_call = '''    def __call__(self, image: Image.Image) -> Image.Image:\n        image_size = image.size\n        input_images = self.transform_image(image).unsqueeze(0).to("cuda")\n        # Prediction\n        with torch.no_grad():\n            preds = self.model(input_images)[-1].sigmoid().cpu()\n'''
    new_call = '''    def __call__(self, image: Image.Image) -> Image.Image:\n        image_size = image.size\n        force_cpu = os.environ.get("REMBG_FORCE_CPU") == "1"\n        device = "cpu" if force_cpu else "cuda"\n        if force_cpu:\n            self.model.cpu()\n        input_images = self.transform_image(image).unsqueeze(0).to(device)\n        # Prediction\n        with torch.no_grad():\n            preds = self.model(input_images)[-1].sigmoid().cpu()\n'''
    if old_call in s:
        s = s.replace(old_call, new_call)
    elif 'REMBG_FORCE_CPU' not in s:
        raise RuntimeError(f'Could not patch BiRefNet __call__ in {p}')
    p.write_text(s)


def main():
    patch_image_feature_extractor()
    patch_birefnet()
    print('Applied TRELLIS.2 local model/rembg override patches')


if __name__ == '__main__':
    main()

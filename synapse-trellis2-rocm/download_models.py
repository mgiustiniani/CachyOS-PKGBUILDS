#!/usr/bin/env python3
"""Mirror-aware model fetch helper for TRELLIS.2 ROCm.

The package/install flow calls this helper to fetch weights into the configured
external cache. It reads the same models.conf that runtime uses and keeps
weights outside packages/ISO images.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def load_conf(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def run(cmd: list[str], env: dict[str, str] | None = None):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def download_url(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f'skip existing {dest}')
        return
    print(f'download {url} -> {dest}', flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open('wb') as f:
        shutil.copyfileobj(r, f)


def fetch_hf(repo_id: str, hf_home: Path):
    env = os.environ.copy()
    env['HF_HOME'] = str(hf_home)
    code = (
        'from huggingface_hub import snapshot_download; '
        f'print(snapshot_download({repo_id!r}))'
    )
    run([sys.executable, '-c', code], env=env)


def fetch_modelscope_dino(url_base: str, dest: Path):
    # ModelScope resolve endpoint used during validation.
    if '/models/' in url_base:
        resolve = url_base.rstrip('/') + '/resolve/master'
    else:
        resolve = url_base.rstrip('/')
    files = ['config.json', 'preprocessor_config.json', 'LICENSE.md', 'README.md', 'model.safetensors']
    for name in files:
        download_url(f'{resolve}/{name}', dest / name)


def copy_tree(src_url: str, dest: Path):
    src = src_url.removeprefix('file://')
    srcp = Path(src)
    if not srcp.exists():
        raise FileNotFoundError(src)
    if dest.exists():
        print(f'skip existing {dest}')
        return
    print(f'copy {srcp} -> {dest}', flush=True)
    shutil.copytree(srcp, dest, symlinks=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', default=os.environ.get('TRELLIS2_MODELS_CONF', '/etc/synapse/trellis2-rocm/models.conf'))
    ap.add_argument('--root', default=None, help='Override TRELLIS2_MODEL_ROOT')
    ap.add_argument('--skip-hf', action='store_true')
    ap.add_argument('--skip-dino', action='store_true')
    args = ap.parse_args()

    conf_path = Path(args.conf)
    if not conf_path.exists():
        local = Path('config/models.conf')
        example = Path('config/models.conf.example')
        conf_path = local if local.exists() else example
    conf = load_conf(conf_path)

    root = Path(args.root or conf.get('TRELLIS2_MODEL_ROOT') or os.path.expanduser('~/models/trellis2'))
    hf_home = Path(conf.get('HF_HOME') or root / 'huggingface')
    print('config:', conf_path)
    print('model root:', root)
    print('HF_HOME:', hf_home)

    if not args.skip_hf:
        fetch_hf(conf.get('TRELLIS2_MODEL_ID', 'microsoft/TRELLIS.2-4B'), hf_home)
        fetch_hf(conf.get('TRELLIS_IMAGE_LARGE_ID', 'microsoft/TRELLIS-image-large'), hf_home)
        fetch_hf(conf.get('REMBG_MODEL_ID', conf.get('REMBG_MODEL_NAME', 'ZhengPeng7/BiRefNet')), hf_home)

    if not args.skip_dino:
        dino_dest = Path(conf.get('DINO_MODEL_PATH') or root / 'dino/facebook/dinov3-vitl16-pretrain-lvd1689m')
        dino_url = conf.get('DINO_MODEL_URL', '')
        if dino_url.startswith('file://'):
            copy_tree(dino_url, dino_dest)
        elif 'modelscope.cn' in dino_url:
            fetch_modelscope_dino(dino_url, dino_dest)
        elif dino_url:
            print('DINO_MODEL_URL is not a built-in downloader type; use file:// or pre-populate DINO_MODEL_PATH:', dino_url)
        else:
            print('No DINO_MODEL_URL set; skipping')


if __name__ == '__main__':
    main()

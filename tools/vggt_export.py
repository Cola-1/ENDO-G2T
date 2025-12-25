import os
import sys
import argparse
import glob
import numpy as np

import torch

# Use StreamVGGT's official exporter
STREAMVGGTPATH = '/root/autodl-tmp/StreamVGGT'
SCRIPT_PATH = os.path.join(STREAMVGGTPATH, 'scripts', 'export_priors.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root containing images dir')
    parser.add_argument('--out_dir', type=str, default='priors/streamvggt')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    seq_dir = os.path.abspath(args.data_root)
    out_dir = os.path.join(seq_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Call StreamVGGT's exporter as a module
    cmd = f"python {SCRIPT_PATH} {seq_dir} --out_dir {out_dir}"
    if args.ckpt is not None:
        cmd += f" --ckpt {args.ckpt}"
    print(f"Running: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError('StreamVGGT export_priors failed')

    print(f'Exported priors to {out_dir}')


if __name__ == '__main__':
    main()



import sys
import os
import time
import json
import numpy as np
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.speed_utils import load_active_indices
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.general_utils import safe_state


def recursive_merge(args, cfg):
    def _merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                _merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        _merge(k, cfg)


@torch.no_grad()
def main():
    parser = ArgumentParser(description="Benchmark FPS/metrics with optional pruning and keyframe filtering")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iteration", default=None, type=int)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "filtered", "pruned", "both"])  # both = pruned+filtered
    parser.add_argument("--keyframe_dir", type=str, default=None)
    parser.add_argument("--prune_mask", type=str, default=None)  # path to prune_keep_mask.npy
    parser.add_argument("--subset", type=str, default="test", choices=["train", "test"])  # which split to evaluate
    # Fill root-level config keys to allow recursive_merge
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exhaust_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    cfg = OmegaConf.load(args.config)
    recursive_merge(args, cfg)

    safe_state(False)

    device = torch.device("cuda")
    gaussians = GaussianModel(args.sh_degree,
                              gaussian_dim=args.gaussian_dim,
                              time_duration=args.time_duration,
                              rot_4d=args.rot_4d,
                              force_sh_3d=args.force_sh_3d,
                              sh_degree_t=2 if pp.extract(args).eval_shfs_4d else 0)
    scene = Scene(lp.extract(args), gaussians, load_iteration=args.iteration, shuffle=False,
                  num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    gaussians.training_setup(op.extract(args))

    views = scene.getTestCameras() if args.subset == "test" else scene.getTrainCameras()

    # Prepare global pruned indices if needed
    pruned_indices = None
    if args.mode in ("pruned", "both") and args.prune_mask is not None and os.path.isfile(args.prune_mask):
        keep_mask = np.load(args.prune_mask)
        keep_mask = torch.from_numpy(keep_mask.astype(np.bool_))
        pruned_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).long().to(device)

    # Prepare keyframe times if needed
    keyframe_times = None
    if args.mode in ("filtered", "both") and args.keyframe_dir is not None and os.path.isdir(args.keyframe_dir):
        times_path = os.path.join(args.keyframe_dir, 'times.npy')
        if os.path.isfile(times_path):
            keyframe_times = np.load(times_path)

    bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    t_start = None
    num_frames = 0
    psnrs = []
    ssims = []

    for idx, (gt_img, view) in enumerate(tqdm(views, desc=f"Benchmark {args.mode}")):
        if t_start is None:
            t_start = time.time()

        view = view.cuda()

        # Decide active indices based on mode
        active_indices = None
        if args.mode in ("filtered", "both") and keyframe_times is not None:
            t = float(view.timestamp)
            pos = int(np.searchsorted(keyframe_times, t))
            left = max(0, pos - 1)
            right = min(len(keyframe_times) - 1, pos)
            merged = None
            for k in sorted(set([left, right])):
                p = os.path.join(args.keyframe_dir, f'idx_{k}.npy')
                if os.path.isfile(p):
                    arr = load_active_indices(p).long().to(device)
                    merged = arr if merged is None else torch.unique(torch.cat([merged, arr], dim=0))
            if merged is not None:
                active_indices = merged

        if args.mode in ("pruned", "both") and pruned_indices is not None:
            if active_indices is None:
                active_indices = pruned_indices
            else:
                # intersection: only keep gaussians that survive pruning and are active in keyframe masks
                active_indices = torch.from_numpy(np.intersect1d(active_indices.detach().cpu().numpy(), pruned_indices.detach().cpu().numpy())).long().to(device)

        if active_indices is not None and active_indices.numel() > 0:
            sub_gaussians = gaussians.subset_view(active_indices)
            rendering = render(view, sub_gaussians, pp.extract(args), background)
        else:
            rendering = render(view, gaussians, pp.extract(args), background)

        image = rendering["render"].clamp(0.0, 1.0)
        gt = gt_img.cuda()
        mask = view.mask
        if mask is not None:
            mask = mask.cuda()
            image = image * mask
            gt = gt * mask

        psnrs.append(psnr(image, gt).mean().double().item())
        ssims.append(ssim(image, gt).mean().double().item())
        num_frames += 1

    elapsed = time.time() - t_start if t_start is not None else 0.0
    fps = num_frames / elapsed if elapsed > 0 else 0.0

    report = {
        "mode": args.mode,
        "subset": args.subset,
        "frames": num_frames,
        "elapsed_sec": elapsed,
        "fps": fps,
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
    }

    out_dir = os.path.join(args.model_path, 'bench_reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bench_{args.mode}_{args.subset}.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\nBench report:")
    print(json.dumps(report, indent=2))
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()



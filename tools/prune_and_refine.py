import sys
import os
import random
import numpy as np
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.speed_utils import ensure_dir, approx_spatial_scores, approx_temporal_scores, rank_and_prune
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
    parser = ArgumentParser(description="Global pruning by approximate SS/ST scoring")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--time_samples", type=int, default=9)
    parser.add_argument("--view_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iteration", default=None, type=int)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    safe_state(False)

    gaussians = GaussianModel(args.sh_degree,
                              gaussian_dim=args.gaussian_dim,
                              time_duration=args.time_duration,
                              rot_4d=args.rot_4d,
                              force_sh_3d=args.force_sh_3d,
                              sh_degree_t=2 if pp.extract(args).eval_shfs_4d else 0)
    scene = Scene(lp.extract(args), gaussians, load_iteration=args.iteration, shuffle=False,
                  num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)

    gaussians.training_setup(op.extract(args))

    N = gaussians.get_xyz.shape[0]
    device = torch.device("cuda")
    vis_count = torch.zeros((N,), dtype=torch.float32, device=device)
    radii_sum = torch.zeros((N,), dtype=torch.float32, device=device)
    mean_opacity = gaussians.get_opacity.detach().to(device).squeeze(-1) if gaussians.get_opacity.dim() > 1 else gaussians.get_opacity.detach().to(device)

    # Sample times uniformly in [t0, t1]
    t0, t1 = args.time_duration
    times = torch.linspace(t0, t1, steps=max(2, args.time_samples), device=device)

    # Sample views from training cameras
    train_views = scene.getTrainCameras()
    sample_indices = list(range(len(train_views)))
    if args.view_samples < len(sample_indices):
        sample_indices = random.sample(sample_indices, k=args.view_samples)

    bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Accumulate visibility and radii statistics
    for ti, t in enumerate(times):
        for si in tqdm(sample_indices, desc=f"Scoring t={float(t):.3f}"):
            gt_img, view = train_views[si]
            view = view.cuda()
            view.timestamp = float(t)
            pkg = render(view, gaussians, pp.extract(args), background)
            visible = pkg["visibility_filter"].to(device)
            radii = pkg["radii"].to(device)
            vis_count += visible.float()
            radii_sum += radii * visible.float()

    mean_radii = radii_sum / (vis_count + 1e-6)

    # Temporal scores via velocity variance across time
    velocity_list = []
    for t in times:
        _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, float(t) + 0.1)
        velocity_list.append(velocity.detach().to(device))

    ss = approx_spatial_scores(vis_count, mean_radii, mean_opacity)
    st = approx_temporal_scores(velocity_list)

    keep_mask = rank_and_prune(ss, st, keep_ratio=args.keep_ratio).cpu().numpy()

    out_dir = os.path.join(args.model_path, 'speed_cache')
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, 'prune_keep_mask.npy'), keep_mask.astype(np.bool_))
    np.save(os.path.join(out_dir, 'ss.npy'), ss.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'st.npy'), st.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 's.npy'), (ss * st).detach().cpu().numpy())

    kept = int(keep_mask.sum())
    print(f"Saved pruning mask: kept {kept}/{N} ({kept / float(N):.2%}) to {out_dir}")


if __name__ == "__main__":
    main()



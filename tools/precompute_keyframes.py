import sys
import os
import numpy as np
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.speed_utils import ensure_dir, save_active_indices, choose_keyframe_times
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


def main():
    parser = ArgumentParser(description="Precompute keyframe active indices")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delta_t", type=float, default=10)
    parser.add_argument("--alpha_thresh", type=float, default=0.01)
    parser.add_argument("--out", type=str, default=None)
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

    if args.out is None:
        out_dir = os.path.join(args.model_path, 'speed_cache')
    else:
        out_dir = args.out
    ensure_dir(out_dir)

    # Build keyframe times from dataset time range
    t_min, t_max = args.time_duration[0], args.time_duration[1]
    times = choose_keyframe_times(t_min, t_max, args.delta_t)
    np.save(os.path.join(out_dir, 'times.npy'), np.asarray(times, dtype=np.float32))

    # Use train cameras for visibility union
    train_views = scene.getTrainCameras()

    bg_color = [1, 1, 1] if lp.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for idx_t, t in enumerate(times):
        active_mask = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
        for gt_img, view in tqdm(train_views, desc=f"Keyframe {idx_t}/{len(times)}"):
            view = view.cuda()
            # overwrite timestamp temporarily
            view.timestamp = float(t)
            pkg = render(view, gaussians, pp.extract(args), background)
            visible = pkg["visibility_filter"].to(device=active_mask.device)
            # approximate using visibility only; threshold alpha additionally if needed in future
            active_mask |= visible

        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1).long().cpu()
        save_active_indices(os.path.join(out_dir, f"idx_{idx_t}.npy"), active_indices)

    print("Keyframe masks saved to:", out_dir)


if __name__ == "__main__":
    main()



#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import platform
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim, SILogLoss, GradL1Loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_rigid_for_log = 0.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None

    gaussians.env_map = env_map

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=12 if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)

    # Geometry prior quick check (visible in logs)
    try:
        sample_has_prior = False
        if isinstance(training_dataset, (list, tuple)) and len(training_dataset) > 0:
            _gt_img0, _cam0 = training_dataset[0]
            sample_has_prior = getattr(_cam0, 'prior_depth', None) is not None
        prior_dir = getattr(dataset, 'vggt_prior_dir', '')
        print(f"[PRIOR CHECK] use_vggt_priors={getattr(dataset, 'use_vggt_priors', False)} | prior_dir='{prior_dir}' exists={os.path.isdir(prior_dir) if isinstance(prior_dir, str) else False} | use_scale_depth={getattr(opt, 'use_scale_depth', False)} | lambda_si={getattr(opt, 'lambda_si', 0.0)} | lambda_depth_grad={getattr(opt, 'lambda_depth_grad', 0.0)} | sample_has_prior={sample_has_prior}")
    except Exception as e:
        print(f"[PRIOR CHECK] failed: {e}")

    iteration = first_iter

    Lrigid = None
    # Optional depth-related losses (default off)
    si_crit = SILogLoss(beta=0.15) if getattr(opt, 'lambda_si', 0.0) > 0 else None
    grad_crit = GradL1Loss() if getattr(opt, 'lambda_depth_grad', 0.0) > 0 else None


    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()

            # Render

            if (iteration - 1) == debug_from:
                pipe.debug = True
                print('debug start')

            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []

            for batch_idx in range(batch_size):
                gt_image, viewpoint_cam = batch_data[batch_idx]

                gt_image = gt_image.cuda()
                viewpoint_cam = viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]

                # Initialize optional logging tensors for priors
                Lsi = torch.tensor(0.0, device=image.device)
                Ldepth_grad = torch.tensor(0.0, device=image.device)


                if viewpoint_cam.mask is not None:
                    mask_tensor = viewpoint_cam.mask.bool().to(image.device)

                    # 规范化mask到[H,W]
                    if mask_tensor.dim() == 4 and mask_tensor.size(1) == 1:
                        mask_2d = mask_tensor[0, 0].bool()
                    elif mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
                        mask_2d = mask_tensor[0].bool()
                    elif mask_tensor.dim() == 2:
                        mask_2d = mask_tensor.bool()
                    else:
                        mask_2d = mask_tensor.squeeze().bool()
                        assert mask_2d.dim() == 2, f"Unexpected mask shape: {mask_tensor.shape}"

                    # 应用到图像（自动广播到通道维）
                    image *= mask_2d
                    gt_image *= mask_2d

                    # 为l1_loss构造4D mask [B,C,H,W]
                    B = 1
                    C, H, W = image.shape
                    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

                    Ll1 = l1_loss(image.unsqueeze(0),
                                  gt_image.unsqueeze(0),
                                  mask=mask_4d)

                else:
                    Ll1 = l1_loss(image.unsqueeze(0), gt_image.unsqueeze(0), mask=None)

                loss = Ll1

                if (viewpoint_cam.mask is not None) and (viewpoint_cam.depth is not None):
                    gt_depth = viewpoint_cam.depth.to(image.device)
                    dep_mask = torch.logical_and(gt_depth > 0, depth > 0)

                    gt_depth = gt_depth * dep_mask
                    depth = depth * dep_mask

                    depth_tensor = depth * mask_2d
                    gt_depth_tensor = gt_depth * mask_2d

                    # Compute normals from depth
                    def compute_normals(depth_tensor):
                        # 接受 [H,W] 或 [1,H,W] 或 [B,1,H,W]，统一到 [H,W]
                        if depth_tensor.dim() == 3 and depth_tensor.size(0) == 1:
                            depth_2d = depth_tensor[0]
                        elif depth_tensor.dim() == 4 and depth_tensor.size(1) == 1:
                            depth_2d = depth_tensor[0, 0]
                        else:
                            depth_2d = depth_tensor

                        # 计算深度梯度 [H,W]
                        dzdx = torch.gradient(depth_2d, dim=1)[0]
                        dzdy = torch.gradient(depth_2d, dim=0)[0]

                        # 创建3D向量场 [3,H,W]，法向量≈(-dzdx, -dzdy, 1)
                        normals = torch.stack([
                            -dzdx,
                            -dzdy,
                            torch.ones_like(dzdx)
                        ], dim=0)

                        # 归一化
                        norm = torch.norm(normals, dim=0, keepdim=True)
                        normals = normals / (norm + 1e-6)

                        return normals  # [3,H,W]

                    # 算 rendered and gt normals
                    rendered_normals = compute_normals(depth_tensor)  # [3, H, W]
                    gt_normals = compute_normals(gt_depth_tensor)     # [3, H, W]

                    # 调整掩码维度以匹配法向量 [3,H,W]
                    mask_3d = mask_2d.unsqueeze(0).expand(3, -1, -1)

                    # ENAC
                    LENAC = l1_loss(rendered_normals, gt_normals, mask=mask_3d)
                    loss += opt.lambda_ENAC * LENAC


                # Hybrid: Original + VGGT Prior (Progressive Integration)
                # 🔍 调试输出：验证几何先验路径
                vggt_check = getattr(dataset, 'use_vggt_priors', False)
                depth_weight_check = getattr(opt, 'prior_depth_weight', 0.0) > 0
                si_check = getattr(opt, 'lambda_si', 0.0) > 0
                grad_check = getattr(opt, 'lambda_depth_grad', 0.0) > 0
                
                if iteration % 500 == 0:  # 每500步输出一次检查
                    print(f"\n🔍 [Iter {iteration}] 几何先验检查:")
                    print(f"  VGGT标志: {vggt_check}")
                    print(f"  深度权重: {depth_weight_check}")
                    print(f"  SI权重: {si_check} (值: {getattr(opt, 'lambda_si', 0.0)})")
                    print(f"  梯度权重: {grad_check} (值: {getattr(opt, 'lambda_depth_grad', 0.0)})")
                
                if (vggt_check or depth_weight_check or si_check or grad_check):
                    prior_dep = getattr(viewpoint_cam, 'prior_depth', None)
                    if prior_dep is not None:
                        prior_dep = prior_dep.to(image.device)
                        
                        # Progressive prior weight schedule
                        warmup_steps = getattr(opt, 'prior_warmup_steps', 1000)
                        max_weight = getattr(opt, 'prior_max_weight', 0.01)
                        
                        if iteration < warmup_steps:
                            # Gradually increase prior influence
                            prior_schedule = min(1.0, iteration / warmup_steps)
                        else:
                            prior_schedule = 1.0
                        
                        # Calculate actual weights with schedule
                        current_lambda_si = getattr(opt, 'lambda_si', 0.0) * prior_schedule * max_weight
                        current_lambda_depth_grad = getattr(opt, 'lambda_depth_grad', 0.0) * prior_schedule * max_weight
                        
                        # 🔍 调试输出：权重计算详情
                        if iteration % 500 == 0:
                            print(f"  📊 权重计算详情:")
                            print(f"    先验调度: {prior_schedule:.3f}")
                            print(f"    最大权重: {max_weight}")
                            print(f"    当前SI权重: {current_lambda_si:.8f}")
                            print(f"    当前梯度权重: {current_lambda_depth_grad:.8f}")
                        
                        # Validity/conf mask (boolean) - more conservative
                        valid_mask = (prior_dep > 0) & (prior_dep < 1.0)  # Filter extreme values
                        prior_conf = getattr(viewpoint_cam, 'prior_conf', None)
                        if prior_conf is not None:
                            # Higher confidence threshold for stability
                            conf_thresh = max(0.01, prior_conf.max().item() * 0.5)
                            valid_mask = torch.logical_and(valid_mask, prior_conf.to(image.device) > conf_thresh)
                        
                        # Apply instrument mask if available
                        if getattr(viewpoint_cam, 'mask', None) is not None:
                            if 'mask_2d' in locals():
                                valid_mask = torch.logical_and(valid_mask, mask_2d.bool())
                        
                        # Only apply priors if enough valid pixels
                        valid_ratio = valid_mask.float().mean()
                        if valid_ratio > 0.1:  # At least 10% valid pixels
                            
                            # 🔍 调试输出：几何先验应用
                            if iteration % 500 == 0:
                                print(f"  🎯 应用几何先验:")
                                print(f"    有效像素比例: {valid_ratio:.3f}")
                                print(f"    先验深度形状: {prior_dep.shape}")
                                print(f"    渲染深度形状: {depth_tensor.shape}")
                            
                            # Scale-invariant depth loss (with safeguards)
                            if si_crit is not None and getattr(opt, 'use_scale_depth', False) and current_lambda_si > 0:
                                try:
                                    pred_d = depth.unsqueeze(0)
                                    prior_d = prior_dep.unsqueeze(0)
                                    
                                    # Normalize depth ranges for stability
                                    pred_d_norm = (pred_d - pred_d.min()) / (pred_d.max() - pred_d.min() + 1e-8)
                                    prior_d_norm = (prior_d - prior_d.min()) / (prior_d.max() - prior_d.min() + 1e-8)
                                    
                                    si_loss = si_crit(pred_d_norm, prior_d_norm, mask=valid_mask)
                                    if torch.isfinite(si_loss):
                                        Lsi = si_loss
                                        loss += current_lambda_si * Lsi
                                        
                                        # 🔍 调试输出：SI损失详情
                                        if iteration % 500 == 0:
                                            print(f"    ✅ SI损失: {Lsi.item():.6f} (权重: {current_lambda_si:.8f})")
                                except Exception as e:
                                    print(f"[SI Loss Error] {e}")
                                    Lsi = torch.tensor(0.0, device=image.device)
                            
                            # Gradient consistency loss (with safeguards)
                            if grad_crit is not None and current_lambda_depth_grad > 0:
                                try:
                                    pred_d = { 'metric_depth': depth.unsqueeze(0) }
                                    prior_d = prior_dep.unsqueeze(0)
                                    grad_loss = grad_crit(pred_d, prior_d, mask=valid_mask)
                                    if torch.isfinite(grad_loss):
                                        Ldepth_grad = grad_loss
                                        loss += current_lambda_depth_grad * Ldepth_grad
                                        
                                        # 🔍 调试输出：梯度损失详情
                                        if iteration % 500 == 0:
                                            print(f"    ✅ 梯度损失: {Ldepth_grad.item():.6f} (权重: {current_lambda_depth_grad:.8f})")
                                except Exception as e:
                                    print(f"[Grad Loss Error] {e}")
                                    Ldepth_grad = torch.tensor(0.0, device=image.device)

                # Loss
                Lssim = 1.0 - ssim(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * loss +  opt.lambda_dssim * Lssim

                if opt.use_depth:
                    depth_weight = opt.depth_weight
                    depth_loss = l1_loss(depth_tensor / (depth_tensor.max() + 1e-6),
                                         gt_depth_tensor / (gt_depth_tensor.max() + 1e-6),
                                         mask=mask_tensor)
                    loss += depth_loss * depth_weight

                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

                if platform.system() == 'Windows' and opt.use_depth and (viewpoint_cam.mask is not None):

                    plt.figure(0)
                    plt.clf()
                    plt.subplot(2, 2, 1)
                    plt.imshow((image.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    plt.subplot(2, 2, 2)
                    plt.imshow((gt_image.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    plt.subplot(2, 2, 3)
                    plt.imshow(depth_tensor.detach().squeeze().cpu().numpy().astype(np.uint8), cmap='gray')
                    plt.subplot(2, 2, 4)
                    plt.imshow(gt_depth_tensor.detach().squeeze().cpu().numpy().astype(np.uint8), cmap='gray')
                    plt.pause(0.0000001)
                    # plt.show()

                ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask

                    Lopa_mask = (- sky * torch.log(1 - o)).mean()

                    # lambda_opa_mask = opt.lambda_opa_mask * (1 - 0.99 * min(1, iteration/opt.iterations))
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask
                ###### opa mask Loss ######

                ###### rigid loss ######
                if opt.lambda_rigid > 0 and iteration >= opt.densify_until_iter:
                    time_start = time.time()
                    k = 20
                    # cur_time = viewpoint_cam.timestamp
                    # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    xyz_mean = gaussians.get_xyz
                    xyz_cur =  xyz_mean #  + delta_mean
                    idx, dist = knn(xyz_cur[None].contiguous().detach(),
                                    xyz_cur[None].contiguous().detach(),
                                    k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)

                    xyz_bef = xyz_cur + velocity

                    time_seed1 = time.time()

                    xyz_cur_dist = torch.norm(xyz_mean[idx] - xyz_mean[None, :, None], p=2, dim=-1)
                    xyz_bef_dist = torch.norm(xyz_bef[idx] - xyz_bef[None, :, None], p=2, dim=-1)

                    time_seed2 = time.time()

                    weight = torch.exp(-100 * dist)

                    adjMat = weight * torch.abs(xyz_cur_dist - xyz_bef_dist) / (xyz_cur_dist + xyz_bef_dist + 1e-10)

                    # Lrigid = torch.exp((adjMat ** 2) / (2 * 10 ** 2)).sum() / (xyz_mean.size(0) * k)
                    Lrigid = adjMat.sum() / (xyz_mean.size(0) * k)
                    time_seed3 = time.time()
                    # print("time1: {:.4f}, time2:{:.4f}, time3:{:.4f}".format(time_seed1-time_start, time_seed2-time_seed1, time_seed3-time_seed2))
                    loss = loss + opt.lambda_rigid * Lrigid
                    # k = 20
                    # # cur_time = viewpoint_cam.timestamp
                    # # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    # xyz_mean = gaussians.get_xyz
                    # xyz_cur =  xyz_mean #  + delta_mean
                    # idx, dist = knn(xyz_cur[None].contiguous().detach(),
                    #                 xyz_cur[None].contiguous().detach(),
                    #                 k)
                    # _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    # weight = torch.exp(-100 * dist)
                    # # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                    # # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                    # # weight *= marginal_weights
                    #
                    # # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                    # # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                    # # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                    # vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    # Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    # loss = loss + opt.lambda_rigid * Lrigid
                ########################
                else:
                    Lrigid = loss

                ###### motion loss ######
                if opt.lambda_motion > 0:
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    Lmotion = velocity.norm(p=2, dim=1).mean()
                    loss = loss + opt.lambda_motion * Lmotion
                ########################

                # 4D-KNN velocity consistency (optional)
                if getattr(opt, 'lambda_knn4d', 0.0) > 0 and gaussians.gaussian_dim == 4:
                    with torch.no_grad():
                        vis_mask = (radii > 0)
                        t_scale = getattr(pipe, 'knn_t_scale', 10.0)
                        t4 = gaussians.get_t.squeeze(-1) * t_scale
                        pts4d = torch.cat([gaussians.get_xyz, t4.unsqueeze(1)], dim=1)[vis_mask]
                        if pts4d.shape[0] >= 2:
                            # 只使用前3个维度(xyz)进行knn计算，忽略时间维度
                            pts3d = pts4d[:, :3]  # 提取xyz坐标
                            
                            # 如果点数太多，跳过KNN计算以避免内存不足
                            max_points_for_knn = 15000  # 限制KNN计算的最大点数
                            if pts3d.shape[0] > max_points_for_knn:
                                print(f"⚠️  跳过KNN计算: 点数 {pts3d.shape[0]} 超过限制 {max_points_for_knn}")
                                idx_knn = None
                            else:
                                idx_knn, _ = knn(pts3d[None].contiguous().detach(), pts3d[None].contiguous().detach(), getattr(opt, 'knn_k', 8))
                        else:
                            idx_knn = None
                    _, vel_all = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    vel_v = vel_all[vis_mask]
                    if idx_knn is not None and vel_v.shape[0] >= 2:
                        vel_nn = vel_v[idx_knn.squeeze(0)]
                        vel_avg = vel_nn.mean(dim=1)
                        loss_knn4d = (vel_avg - vel_v).abs().sum(dim=-1).mean()
                        # Align with logging convention expecting L{term}
                        Lknn4d = loss_knn4d
                        loss = loss + opt.lambda_knn4d * loss_knn4d

                # Opacity entropy regularizer (optional, lightweight)
                if getattr(opt, 'lambda_opa_entropy', 0.0) > 0:
                    o = alpha.clamp(1e-3, 1-1e-3)
                    loss_opa_ent = -(o * torch.log(o)).mean()
                    # Align with logging convention expecting L{term}
                    Lopa_entropy = loss_opa_ent
                    loss = loss + opt.lambda_opa_entropy * loss_opa_ent

                loss = loss / batch_size
                loss.backward()
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]

                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()

            iter_end.record()
            # 将可选的 LENAC 规范成 Python float，便于日志与打印
            if 'LENAC' in locals() and isinstance(LENAC, torch.Tensor):
                lenac_val = float(LENAC.detach().item())
            else:
                lenac_val = 0.0

            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim,
                         "Ldepth": depth_loss,
                          "LENAC": lenac_val,
                          }
            # expose prior-related losses for logging if enabled
            if isinstance(Lsi, torch.Tensor):
                loss_dict["Lsi"] = Lsi
            if isinstance(Ldepth_grad, torch.Tensor):
                loss_dict["Ldepth_grad"] = Ldepth_grad

            with torch.no_grad():
                psnr_for_log = psnr(image, gt_image).mean().double()
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
                # also maintain ema for prior losses if present
                if isinstance(Lsi, torch.Tensor):
                    ema_Lsi_for_log = 0.4 * float(Lsi.item()) + 0.6 * vars().get('ema_Lsi_for_log', 0.0)
                if isinstance(Ldepth_grad, torch.Tensor):
                    ema_Ldepth_grad_for_log = 0.4 * float(Ldepth_grad.item()) + 0.6 * vars().get('ema_Ldepth_grad_for_log', 0.0)
                if Lrigid is not None:
                    ema_rigid_for_log = 0.4 * Lrigid.item() + 0.6 * ema_rigid_for_log
                else:
                    pass

                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]

                if iteration % 10 == 0:
                    elapsed_ms = iter_start.elapsed_time(iter_end)
                    fps = 1000.0 / max(elapsed_ms, 1e-3)
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}",
                                            "Ldepth": f"{ema_depth_for_log:.{4}f}",
                                             "FPS": f"{fps:.{2}f}",
                                             "Lrigid": f"{ema_rigid_for_log:.{4}f}" if Lrigid is not None else f"{0:.{4}f}"}
                    if isinstance(Lsi, torch.Tensor):
                        postfix["Lsi"] = f"{float(Lsi.item()):.{4}f}"
                    if isinstance(Ldepth_grad, torch.Tensor):
                        postfix["Ldepth_grad"] = f"{float(Ldepth_grad.item()):.{4}f}"

                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"

                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_dict)
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                    else:
                        gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)
                    # print(gaussians._xyz.size())
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        env_map_optimizer.step()
                        env_map_optimizer.zero_grad(set_to_none = True)

    # Save final checkpoint to chkpnt_best.pth
    try:
        best_ckpt_path = os.path.join(scene.model_path, "chkpnt_best.pth")
        torch.save((gaussians.capture(), iteration), best_ckpt_path)
        print(f"Saved final checkpoint: {best_ckpt_path}")
    except Exception as e:
        print(f"Warning: failed to save final checkpoint: {e}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'], iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'], iteration)
            if "LENAC" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/ENAC_loss', loss_dict['LENAC'], iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):

                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    mask = viewpoint.mask.cuda().bool()

                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    if mask is not None:
                        mask = mask.cuda()
                        image = image * mask
                        gt_image = gt_image * mask
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                if config['name'] == 'test':
                    ts = time.strftime("%m/%d %H:%M:%S", time.localtime())
                    try:
                        psnr_val = float(psnr_test.item())
                    except Exception:
                        psnr_val = float(psnr_test)
                    print(f"[{iteration}] Evaluating test: PSNR {psnr_val} [{ts}]")
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None )
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=None)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)


    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

    # Print prior-related config summary to logs for verification
    try:
        print(f"[CFG PRIOR] use_vggt_priors={getattr(args, 'use_vggt_priors', False)} | vggt_prior_dir='{getattr(args, 'vggt_prior_dir', '')}' | use_scale_depth={getattr(args, 'use_scale_depth', False)} | lambda_si={getattr(args, 'lambda_si', 0.0)} | lambda_depth_grad={getattr(args, 'lambda_depth_grad', 0.0)}")
    except Exception:
        pass

    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0, op.iterations, 500)]
    if args.eval_interval is not None and args.eval_interval > 0:
        args.test_iterations = list(sorted(set(args.test_iterations + [i for i in range(args.eval_interval, op.iterations + 1, args.eval_interval)])))
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)

    # All done
    print("\nTraining complete.")
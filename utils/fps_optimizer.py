# -*- coding: utf-8 -*-
"""
1000+ FPS优化器 - 实现全局裁剪和关键帧时域过滤
基于论文: "1000+ FPS 4D Gaussian Splatting"

核心技术:
1. 空间-时间变化分数(STV)驱动的全局裁剪
2. 关键帧驱动的时域过滤
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional
import bisect
from tqdm import tqdm
import time


class FPSOptimizer:
    """1000+ FPS优化器主类"""
    
    def __init__(self, 
                 alpha_thresh: float = 0.01,
                 kf_interval: int = 20,
                 prune_ratio: float = 0.80,
                 device: str = "cuda"):
        """
        Args:
            alpha_thresh: 可见性阈值，用于标记激活高斯
            kf_interval: 关键帧间隔
            prune_ratio: 全局裁剪比例 (0.0-0.95)
            device: 计算设备
        """
        self.alpha_thresh = alpha_thresh
        self.kf_interval = kf_interval
        self.prune_ratio = prune_ratio
        self.device = device
        
        # 缓存
        self.keyframes = None
        self.active_sets = None
        self.union_model_cache = {}
        self.union_mask_cache = {}
        
    def compute_spatial_score(self, gaussians, views, pipeline, background, sample_ratio=0.2):
        """
        计算空间分数 SS_i - 基于像素贡献的累积
        
        Args:
            gaussians: 高斯模型
            views: 视角列表
            pipeline: 渲染管线
            background: 背景颜色
            sample_ratio: 采样比例，用于加速计算
            
        Returns:
            spatial_scores: 空间分数张量 [N]
        """
        print("[STV] Computing spatial scores...")
        N = gaussians.get_xyz.shape[0]
        spatial_scores = torch.zeros(N, dtype=torch.float32, device=self.device)
        
        # 采样视角以加速计算
        sample_num = max(1, int(len(views) * sample_ratio))
        sampled_views = views[::max(1, len(views) // sample_num)]
        
        with torch.no_grad():
            for gt_img, view in tqdm(sampled_views, desc="Computing spatial scores"):
                try:
                    # 渲染并获取可见性信息
                    from gaussian_renderer import render
                    rendering = render(view.cuda(), gaussians, pipeline, background)
                    
                    # 获取alpha和可见性过滤器
                    if "alpha" in rendering:
                        alpha = rendering["alpha"].detach()
                        if alpha.dim() == 3:  # [H, W, 1]
                            alpha = alpha.squeeze(-1)  # [H, W]
                        
                        # 累积每个高斯的像素贡献
                        # 这里简化为使用可见性过滤器的频次作为空间分数
                        if "visibility_filter" in rendering:
                            vis_filter = rendering["visibility_filter"].detach()
                            if vis_filter.numel() <= N:
                                spatial_scores[:vis_filter.numel()] += vis_filter.float()
                    
                except Exception as e:
                    print(f"Warning: Failed to process view, skipping. Error: {e}")
                    continue
        
        # 归一化
        spatial_scores = spatial_scores / (len(sampled_views) + 1e-8)
        return spatial_scores
    
    def compute_temporal_score(self, gaussians, time_steps=50):
        """
        计算时间分数 ST_i - 基于时间不透明度的二阶导数
        
        Args:
            gaussians: 高斯模型
            time_steps: 时间步数
            
        Returns:
            temporal_scores: 时间分数张量 [N]
        """
        print("[STV] Computing temporal scores...")
        N = gaussians.get_xyz.shape[0]
        
        # 检查是否有时间相关的参数
        if not hasattr(gaussians, '_t') or gaussians._t.numel() == 0:
            print("Warning: No temporal parameters found, using uniform temporal scores")
            return torch.ones(N, dtype=torch.float32, device=self.device)
        
        # 获取时间参数
        t_means = gaussians._t.detach()  # 时间均值
        
        # 如果有时间协方差参数
        if hasattr(gaussians, '_scaling_t') and gaussians._scaling_t.numel() > 0:
            t_vars = torch.exp(gaussians._scaling_t.detach())  # 时间方差
        else:
            t_vars = torch.ones_like(t_means) * 0.1  # 默认方差
        
        temporal_scores = torch.zeros(N, dtype=torch.float32, device=self.device)
        
        # 在时间范围内采样
        time_range = gaussians.time_duration if hasattr(gaussians, 'time_duration') else [-0.5, 0.5]
        t_samples = torch.linspace(time_range[0], time_range[1], time_steps, device=self.device)
        
        for t in t_samples:
            # 计算时间不透明度 p_i(t)
            p_t = torch.exp(-0.5 * ((t - t_means) / (t_vars + 1e-8)) ** 2)
            
            # 计算二阶导数 (简化版本)
            dt = (time_range[1] - time_range[0]) / time_steps
            second_deriv = ((t - t_means) ** 2 / (t_vars ** 2 + 1e-8) - 1 / (t_vars + 1e-8)) * p_t
            
            # 累积时间变化度 (使用tanh归一化)
            stv_contribution = 0.5 * torch.tanh(torch.abs(second_deriv)) + 0.5
            temporal_scores += stv_contribution * dt
        
        return temporal_scores
    
    def compute_stv_scores(self, gaussians, views, pipeline, background):
        """
        计算综合的空间-时间变化分数
        
        Returns:
            stv_scores: 综合分数张量 [N]
        """
        print("[STV] Computing Space-Time Variance scores...")
        
        # 计算空间分数
        spatial_scores = self.compute_spatial_score(gaussians, views, pipeline, background)
        
        # 计算时间分数  
        temporal_scores = self.compute_temporal_score(gaussians)
        
        # 综合分数 (简化版本的公式7)
        stv_scores = spatial_scores * temporal_scores
        
        print(f"[STV] Spatial scores range: [{spatial_scores.min():.6f}, {spatial_scores.max():.6f}]")
        print(f"[STV] Temporal scores range: [{temporal_scores.min():.6f}, {temporal_scores.max():.6f}]")
        print(f"[STV] Combined STV scores range: [{stv_scores.min():.6f}, {stv_scores.max():.6f}]")
        
        return stv_scores
    
    def global_prune(self, gaussians, views, pipeline, background, opt):
        """
        全局裁剪：基于STV分数删除低贡献的高斯
        
        Returns:
            keep_indices: 保留的高斯索引
        """
        print(f"[Prune] Starting global pruning with ratio {self.prune_ratio}...")
        
        N = gaussians.get_xyz.shape[0]
        print(f"[Prune] Original Gaussians count: {N}")
        
        # 计算STV分数
        stv_scores = self.compute_stv_scores(gaussians, views, pipeline, background)
        
        # 根据分数排序并选择保留的高斯
        keep_num = int((1.0 - self.prune_ratio) * N)
        keep_num = max(keep_num, 1000)  # 至少保留1000个高斯
        
        _, keep_indices = torch.topk(stv_scores, k=keep_num, largest=True)
        keep_indices = keep_indices.sort().values  # 保持原有顺序
        
        print(f"[Prune] Keeping {keep_num}/{N} gaussians ({keep_num/N*100:.1f}%)")
        
        return keep_indices
    
    def build_keyframe_active_sets(self, views, gaussians, pipeline, background):
        """
        构建关键帧激活集合
        
        Returns:
            keyframes: 关键帧列表 [(idx, timestamp), ...]
            active_sets: 激活集合字典 {kf_id: bool_mask}
        """
        print(f"[KF] Building keyframe active sets with interval {self.kf_interval}...")
        
        keyframes = []
        active_sets = {}
        N = gaussians.get_xyz.shape[0]
        
        # 选择关键帧
        for idx, (gt_img, view) in enumerate(views):
            if idx % self.kf_interval == 0:
                timestamp = getattr(view, "timestamp", idx / len(views))
                keyframes.append((idx, timestamp))
        
        print(f"[KF] Selected {len(keyframes)} keyframes from {len(views)} total views")
        
        # 为每个关键帧计算激活集合
        for kf_idx, (idx, timestamp) in enumerate(tqdm(keyframes, desc="Computing active sets")):
            try:
                from gaussian_renderer import render
                view = views[idx][1]
                rendering = render(view.cuda(), gaussians, pipeline, background)
                
                # 获取可见性过滤器
                vis_filter = rendering.get("visibility_filter", torch.zeros(N, dtype=torch.bool, device=self.device))
                
                # 创建激活掩码
                active = torch.zeros(N, dtype=torch.bool, device=self.device)
                if vis_filter.numel() > 0:
                    active[:min(N, vis_filter.numel())] = vis_filter[:min(N, vis_filter.numel())].bool()
                
                # 可选：添加alpha阈值过滤
                if "alpha" in rendering:
                    alpha = rendering["alpha"].detach()
                    if alpha.dim() == 3:
                        alpha_active = (alpha > self.alpha_thresh).any(dim=0).any(dim=0)
                        if alpha_active.numel() <= N:
                            active[:alpha_active.numel()] |= alpha_active
                
                active_sets[kf_idx] = active.cpu()  # 移到CPU节省显存
                
            except Exception as e:
                print(f"Warning: Failed to process keyframe {kf_idx}, using empty set. Error: {e}")
                active_sets[kf_idx] = torch.zeros(N, dtype=torch.bool)
        
        self.keyframes = keyframes
        self.active_sets = active_sets
        
        print(f"[KF] Built active sets for {len(active_sets)} keyframes")
        return keyframes, active_sets
    
    def get_union_active_set(self, frame_idx: int):
        """
        获取指定帧的联合激活集合（基于最近的两个关键帧）
        
        Args:
            frame_idx: 当前帧索引
            
        Returns:
            union_mask: 联合激活掩码
        """
        if self.keyframes is None or self.active_sets is None:
            return None
        
        # 找到最近的两个关键帧
        kf_indices = [kf_idx for kf_idx, (idx, _) in enumerate(self.keyframes)]
        kf_frame_indices = [idx for idx, _ in self.keyframes]
        
        # 使用二分查找找到位置
        pos = bisect.bisect_left(kf_frame_indices, frame_idx)
        left_kf = max(0, pos - 1)
        right_kf = min(len(self.keyframes) - 1, pos)
        
        # 如果左右是同一个关键帧，尝试扩展
        if left_kf == right_kf:
            if left_kf > 0:
                left_kf -= 1
            elif right_kf < len(self.keyframes) - 1:
                right_kf += 1
        
        # 获取联合掩码
        cache_key = (left_kf, right_kf)
        if cache_key in self.union_mask_cache:
            return self.union_mask_cache[cache_key].to(self.device)

        left_mask = self.active_sets.get(left_kf, torch.zeros(0, dtype=torch.bool))
        right_mask = self.active_sets.get(right_kf, torch.zeros(0, dtype=torch.bool))
        
        # 确保掩码长度一致
        max_len = max(left_mask.numel(), right_mask.numel())
        if max_len == 0:
            return None
            
        if left_mask.numel() < max_len:
            padded = torch.zeros(max_len, dtype=torch.bool)
            padded[:left_mask.numel()] = left_mask
            left_mask = padded
            
        if right_mask.numel() < max_len:
            padded = torch.zeros(max_len, dtype=torch.bool)
            padded[:right_mask.numel()] = right_mask
            right_mask = padded
        
        union_mask = left_mask | right_mask
        self.union_mask_cache[cache_key] = union_mask
        return union_mask.to(self.device)
    
    def save_cache(self, cache_path: str):
        """保存关键帧缓存到文件"""
        cache_data = {
            'keyframes': self.keyframes,
            'active_sets': self.active_sets,
            'alpha_thresh': self.alpha_thresh,
            'kf_interval': self.kf_interval
        }
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"[Cache] Saved keyframe cache to {cache_path}")
    
    def load_cache(self, cache_path: str) -> bool:
        """从文件加载关键帧缓存"""
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.keyframes = cache_data['keyframes']
            self.active_sets = cache_data['active_sets']
            
            print(f"[Cache] Loaded keyframe cache from {cache_path}")
            print(f"[Cache] {len(self.keyframes)} keyframes, interval={cache_data.get('kf_interval', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"[Cache] Failed to load cache: {e}")
            return False


def create_subset_gaussian_model(base_model, keep_indices, factory_fn, opt):
    """
    创建高斯模型的子集
    
    Args:
        base_model: 原始高斯模型
        keep_indices: 要保留的高斯索引
        factory_fn: 用于创建新模型的工厂函数
        opt: 优化参数
        
    Returns:
        subset_model: 子集高斯模型
    """
    # 获取原模型状态
    full_state = base_model.capture()
    
    # 创建子集状态
    subset_state = {}
    N_full = base_model.get_xyz.shape[0]
    N_keep = keep_indices.numel()
    
    for key, value in full_state.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == N_full:
            # 这是per-Gaussian的张量，需要子集化
            subset_state[key] = value[keep_indices]
        else:
            # 其他参数保持不变
            subset_state[key] = value
    
    # 创建新模型并恢复状态
    subset_model = factory_fn()
    subset_model.training_setup(opt)
    subset_model.restore(subset_state, opt)
    
    print(f"[Subset] Created subset model: {N_keep}/{N_full} gaussians ({N_keep/N_full*100:.1f}%)")
    
    return subset_model


def benchmark_fps(render_func, views, num_warmup=10, num_test=100):
    """
    FPS基准测试
    
    Args:
        render_func: 渲染函数
        views: 视角列表
        num_warmup: 预热次数
        num_test: 测试次数
        
    Returns:
        fps: 平均FPS
        render_time: 平均渲染时间(ms)
    """
    print(f"[Benchmark] Starting FPS benchmark...")
    print(f"[Benchmark] Warmup: {num_warmup} frames, Test: {num_test} frames")
    
    # 预热
    for i in range(min(num_warmup, len(views))):
        _, view = views[i % len(views)]
        render_func(view.cuda())
    
    # 同步GPU
    torch.cuda.synchronize()
    
    # 测试
    start_time = time.time()
    
    for i in range(num_test):
        _, view = views[i % len(views)]
        render_func(view.cuda())
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_test / total_time
    avg_render_time = (total_time / num_test) * 1000  # ms
    
    print(f"[Benchmark] Results: {fps:.1f} FPS, {avg_render_time:.2f} ms/frame")
    
    return fps, avg_render_time

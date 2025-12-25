"""
pointops2的简化回退实现
当CUDA扩展编译失败时使用的纯PyTorch实现
"""

import torch
import numpy as np


def furthestsampling_fallback(xyz, offset, new_offset):
    """
    Farthest Point Sampling的简化实现
    
    Args:
        xyz: (n, 3) 点云坐标
        offset: (b,) 批次偏移
        new_offset: (b,) 新的批次偏移
    
    Returns:
        idx: (m,) 采样点的索引
    """
    device = xyz.device
    n = xyz.shape[0]
    b = offset.shape[0]
    m = new_offset[-1].item()
    
    # 初始化结果
    idx = torch.zeros(m, dtype=torch.int32, device=device)
    
    current_idx = 0
    for batch_i in range(b):
        # 获取当前批次的点
        start_idx = offset[batch_i-1].item() if batch_i > 0 else 0
        end_idx = offset[batch_i].item()
        batch_xyz = xyz[start_idx:end_idx]
        
        # 当前批次需要采样的点数
        target_start = new_offset[batch_i-1].item() if batch_i > 0 else 0
        target_end = new_offset[batch_i].item()
        n_sample = target_end - target_start
        
        if n_sample <= 0:
            continue
            
        batch_size = batch_xyz.shape[0]
        if batch_size <= n_sample:
            # 如果点数不够，直接使用所有点
            batch_idx = torch.arange(batch_size, device=device)
            # 如果还不够，重复一些点
            if batch_size < n_sample:
                repeat_times = (n_sample + batch_size - 1) // batch_size
                batch_idx = batch_idx.repeat(repeat_times)[:n_sample]
        else:
            # 使用简化的最远点采样
            batch_idx = torch.zeros(n_sample, dtype=torch.long, device=device)
            
            # 选择第一个点（随机或固定）
            batch_idx[0] = 0
            
            # 计算距离并选择后续点
            for i in range(1, n_sample):
                # 计算到已选点的最小距离
                selected_points = batch_xyz[batch_idx[:i]]  # (i, 3)
                distances = torch.cdist(batch_xyz, selected_points)  # (n, i)
                min_distances = distances.min(dim=1)[0]  # (n,)
                
                # 选择距离最远的点
                batch_idx[i] = min_distances.argmax()
        
        # 转换为全局索引
        idx[current_idx:current_idx + n_sample] = batch_idx[:n_sample] + start_idx
        current_idx += n_sample
    
    return idx


def knnquery_fallback(nsample, xyz, new_xyz, offset, new_offset):
    """
    KNN查询的简化实现
    
    Args:
        nsample: 每个查询点的邻居数量
        xyz: (n, 3) 原始点云
        new_xyz: (m, 3) 查询点云
        offset: (b,) 原始点云批次偏移
        new_offset: (b,) 查询点云批次偏移
    
    Returns:
        idx: (m, nsample) 邻居索引
        dist: (m, nsample) 邻居距离
    """
    if new_xyz is None:
        new_xyz = xyz
        
    device = xyz.device
    m = new_xyz.shape[0]
    b = offset.shape[0]
    
    idx = torch.zeros(m, nsample, dtype=torch.int32, device=device)
    dist = torch.zeros(m, nsample, dtype=torch.float32, device=device)
    
    current_query_idx = 0
    for batch_i in range(b):
        # 获取当前批次的原始点
        src_start = offset[batch_i-1].item() if batch_i > 0 else 0
        src_end = offset[batch_i].item()
        batch_xyz = xyz[src_start:src_end]
        
        # 获取当前批次的查询点
        query_start = new_offset[batch_i-1].item() if batch_i > 0 else 0
        query_end = new_offset[batch_i].item()
        batch_new_xyz = new_xyz[query_start:query_end]
        
        n_query = query_end - query_start
        if n_query <= 0:
            continue
        
        # 计算距离矩阵
        distances = torch.cdist(batch_new_xyz, batch_xyz)  # (n_query, n_src)
        
        # 对每个查询点找到最近的nsample个邻居
        for q_idx in range(n_query):
            query_distances = distances[q_idx]  # (n_src,)
            
            # 如果源点数量少于nsample，重复使用
            n_src = batch_xyz.shape[0]
            if n_src < nsample:
                # 重复索引
                sorted_indices = torch.argsort(query_distances)
                repeated_indices = sorted_indices.repeat((nsample + n_src - 1) // n_src)[:nsample]
                batch_idx = repeated_indices
                batch_dist = query_distances[repeated_indices]
            else:
                # 正常情况：选择最近的nsample个点
                topk_result = torch.topk(query_distances, nsample, largest=False)
                batch_dist = topk_result.values
                batch_idx = topk_result.indices
            
            # 转换为全局索引
            global_query_idx = current_query_idx + q_idx
            idx[global_query_idx] = batch_idx + src_start
            dist[global_query_idx] = batch_dist
        
        current_query_idx += n_query
    
    return idx, dist


# 创建与原始API兼容的函数
def furthestsampling(xyz, offset, new_offset):
    """兼容原始API的最远点采样"""
    return furthestsampling_fallback(xyz, offset, new_offset)


def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    """兼容原始API的KNN查询"""
    return knnquery_fallback(nsample, xyz, new_xyz, offset, new_offset)

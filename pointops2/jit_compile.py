#!/usr/bin/env python3
"""
JIT编译pointops2 CUDA扩展的脚本
解决pointops2_cuda模块导入问题
"""

import os
import torch
from torch.utils.cpp_extension import load
import glob

def compile_pointops2():
    """使用JIT编译pointops2扩展"""
    
    print("🔧 开始JIT编译pointops2 CUDA扩展...")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    # 收集所有源文件
    cpp_files = []
    cu_files = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.cpp'):
                cpp_files.append(os.path.join(root, file))
            elif file.endswith('.cu'):
                cu_files.append(os.path.join(root, file))
    
    sources = cpp_files + cu_files
    
    print(f"📁 找到 {len(cpp_files)} 个 .cpp 文件")
    print(f"📁 找到 {len(cu_files)} 个 .cu 文件")
    
    # 编译参数
    extra_cflags = ['-O2']
    extra_cuda_cflags = ['-O2']
    
    # 包含目录
    include_dirs = [src_dir]
    
    try:
        # JIT编译
        print("⚡ 正在进行JIT编译...")
        pointops2_cuda = load(
            name="pointops2_cuda",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=include_dirs,
            verbose=True
        )
        
        print("✅ pointops2_cuda 编译成功!")
        return pointops2_cuda
        
    except Exception as e:
        print(f"❌ JIT编译失败: {e}")
        return None


def test_compilation():
    """测试编译结果"""
    try:
        import pointops2_cuda
        print("✅ pointops2_cuda 导入成功!")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


if __name__ == "__main__":
    # 尝试编译
    module = compile_pointops2()
    
    if module:
        print("\n🎉 编译完成!")
        
        # 测试导入
        if test_compilation():
            print("🎯 pointops2 CUDA扩展已准备就绪!")
        else:
            print("⚠️  编译成功但导入失败，可能需要重启Python环境")
    else:
        print("\n💡 建议尝试以下解决方案:")
        print("1. 检查CUDA和PyTorch版本兼容性")
        print("2. 确保有足够的编译时间和内存")
        print("3. 尝试降低编译并发数: MAX_JOBS=1")

结论先说在前面：
**把“1000+ FPS”的核心（关键帧掩码过滤 + 全局裁剪）落到你的工程里，优先改 `render` 路径。**训练环节可以不动；如果你想把“全局裁剪后的小幅微调”也自动化，才需要在 `train` 里加一个很短的**后处理微调**分支，但不是必需。下面给你可直接落地的改法与原因（含补丁代码，最小侵入），并明确指出每一步放在哪个文件。

* **必须改的位置：`render.py`** → 实现**关键帧掩码过滤（时域局部筛选）**，渲染时只对“激活高斯集合”做 raster。这是 1000+ FPS 的主要来源。
* **可选后处理：放在 `render.py` 或做个单独脚本** → 实现**全局裁剪（时空打分 + 剪枝）**，把短寿命/低贡献的 4D 高斯整体删掉，然后（可选）再用 `train1.py` 做个 **5k–10k 次的微调**把质量补回来。

---

# 你该怎么改（可直接贴的补丁思路）

下面这套改法完全基于你现有接口，**不要求你改 `gaussian_renderer.render()` 的签名**，也**不改你的训练主循环**。做法是：在渲染前，把“激活高斯集合”剪出来，构建一个**子模型**（subset 的 `GaussianModel`），然后用这个子模型去 `render()`。为了避免每一帧都建子模型，我们**按关键帧对构建并缓存**（一段时间只用一个 union 子模型），这样额外开销极小。代码挂载在 `render.py` 内，见下述补丁位点。

## A. 在 `render.py` 顶部加一些参数

（放到 argparse 处与现有参数一起）

```python
parser.add_argument("--use_kf_filter", action="store_true",
                    help="Enable keyframe-based active-Gaussian filtering for 1000+ FPS.")
parser.add_argument("--kf_interval", type=int, default=20,
                    help="Keyframe interval (frames).")
parser.add_argument("--alpha_thresh", type=float, default=0.01,
                    help="Visibility threshold to mark a Gaussian 'active' in a keyframe.")
parser.add_argument("--use_global_prune", action="store_true",
                    help="Run a one-off global prune (post-training) before rendering.")
parser.add_argument("--prune_ratio", type=float, default=0.80,
                    help="Fraction of Gaussians to prune globally (0.0-0.95 recommended).")
```

> 说明：`--use_kf_filter` 打开“关键帧过滤”；`--use_global_prune` 打开“一次性全局裁剪”。你可以单独开其中之一，也可以两个都开（论文里两者叠加能拿到最大提速）。

## B. 在 `render.py` 里加三个工具函数

放在文件内（比如 `render_sets` 上面），用于：
1）**统计关键帧掩码**（用 `render()` 返回的 `visibility_filter`），
2）**从基模型构建子模型**（subset `GaussianModel`），
3）**把相邻关键帧的掩码做并集并缓存**（按时间段复用）。

```python
import torch

def _capture_state(model):
    # 利用你已有的 state 存取接口（训练/测试都在用）
    return model.capture()

def _subset_state(state: dict, keep_idx: torch.Tensor):
    """对 state 中‘以 N 为第一维’的张量做索引子集，其他不动。"""
    # 找到 per-Gaussian 维度 N（取所有张量第一维的最大值来近似）
    N = 0
    for v in state.values():
        if torch.is_tensor(v) and v.dim() > 0:
            N = max(N, v.shape[0])
    sub = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
            sub[k] = v[keep_idx]
        else:
            sub[k] = v
    return sub

def _build_subset_model(base_model, opt, factory_fn, keep_idx: torch.Tensor):
    """从基模型裁一个子模型（只保留 keep_idx 对应的高斯）"""
    full = _capture_state(base_model)
    sub  = _subset_state(full, keep_idx)
    gm = factory_fn()
    gm.training_setup(opt)
    gm.restore(sub, opt)
    return gm

@torch.no_grad()
def build_keyframe_active_sets(views, gaussians, pipeline, background, kf_interval=20, alpha_thresh=0.01):
    """
    为若干关键帧计算 ‘激活高斯集合’（布尔向量），集合 = 该帧下 visibility_filter | (alpha>阈值) 的并。
    用 view.timestamp（若无则用帧序号）来确定时间位置。
    """
    keyframes = []
    active_sets = {}  # key: kf_id -> bool mask [N]
    N = gaussians.get_xyz.shape[0]  # 每个场景的高斯数量（per-point维度）
    for idx, (gt_img, view) in enumerate(views):
        t = getattr(view, "timestamp", idx)
        if idx % kf_interval == 0:
            keyframes.append((idx, t))

    for kf_idx, (idx, t) in enumerate(keyframes):
        # 用当前帧渲染一次，拿 visibility_filter（谁真的参与了这帧合成）
        pkg = render(views[idx][1].cuda(), gaussians, pipeline, background)
        vis = pkg["visibility_filter"].detach().to(torch.bool).cpu()
        # 也可以把 alpha 阈值一起并进去（某些实现里 visibility_filter 已经很干净，这一步可选）
        alpha = pkg["alpha"].detach()
        active = vis.clone()
        if alpha is not None:
            active |= (alpha > alpha_thresh).any() if alpha.dim()==3 else (alpha > alpha_thresh)
        # 对齐到长度 N
        if active.numel() != N:
            a = torch.zeros(N, dtype=torch.bool)
            a[:min(N, active.numel())] = active[:min(N, active.numel())]
            active = a
        active_sets[kf_idx] = active
    return keyframes, active_sets

def union_subset_cache_factory(base_model, opt, factory_fn, active_sets):
    """
    返回一个取并集子模型的 ‘缓存工厂’：
    给定相邻关键帧索引 (i, i+1) -> 返回/缓存 这对 keyframe 的 union 子模型。
    """
    cache = {}
    def get_union_model(i, j):
        key = (i, j)
        if key in cache:
            return cache[key]
        keep = active_sets[i] | active_sets[j]
        keep_idx = torch.nonzero(keep).view(-1).to(base_model.get_xyz.device)
        sub = _build_subset_model(base_model, opt, factory_fn, keep_idx)
        cache[key] = sub
        return sub
    return get_union_model
```

## C. 在 `render_sets()` 里接入“全局裁剪（可选）”与“关键帧过滤（推荐）”

你现有 `render_sets()` 已经把场景和 `GaussianModel` 创建好并 `restore` 了，我们直接在这里加两段逻辑（不破坏原有路径）：

```python
def render_sets(...):
    with torch.no_grad():
        gaussians = GaussianModel(...); scene = Scene(...); gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        # 1) Optional: 全局裁剪（一次性）
        if args.use_global_prune:
            print("[Prune] Global pruning starts ...")
            # 一个轻量级近似：按 ‘训练集各帧 visibility_filter 被激活的频率’ 作为空间分数，
            # 再乘一个时间平滑项（相邻帧激活IoU）作为时间分数，得打分 S_i 排序。
            N = gaussians.get_xyz.shape[0]
            act_count = torch.zeros(N, dtype=torch.float32, device="cuda")
            # 遍历一小撮训练帧（可抽样）统计激活频率
            train_views = scene.getTrainCameras()
            step = max(1, len(train_views)//50)  # 抽50帧左右即可
            for k,(gt_img, v) in enumerate(train_views[::step]):
                pkg = render(v.cuda(), gaussians, pipeline, background)
                vis = pkg["visibility_filter"].float()
                act_count[:vis.numel()] += vis.to(act_count.dtype)

            # 简易时间平滑：前后两帧vis的 IoU 平均（越稳越高）
            smooth = torch.zeros_like(act_count)
            prev = None
            for k,(gt_img, v) in enumerate(train_views[::step]):
                pkg = render(v.cuda(), gaussians, pipeline, background)
                vis = pkg["visibility_filter"].to(torch.bool)
                if prev is not None:
                    inter = (vis & prev).float().sum()
                    union = (vis | prev).float().sum().clamp_min(1.0)
                    s = (inter/union)
                    smooth[:vis.numel()] += s
                prev = vis
            score = act_count * (1.0 + smooth)  # 简单联合分数（足以决定排序）
            keep_num = int((1.0 - args.prune_ratio) * N)
            keep_idx = torch.topk(score, k=keep_num, largest=True).indices
            # 用 subset 子模型替换大模型
            def factory_fn():
                return GaussianModel(scene.dataset.sh_degree, gaussian_dim=gaussians.gaussian_dim,
                                     time_duration=scene.time_duration, rot_4d=gaussians.rot_4d,
                                     force_sh_3d=gaussians.force_sh_3d,
                                     sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
            gaussians = _build_subset_model(gaussians, opt, factory_fn, keep_idx)
            print(f"[Prune] Kept {keep_num}/{N} gaussians.")

        # 2) Keyframe 过滤（强烈推荐开启）
        get_union_model = None
        keyframes = None
        if args.use_kf_filter:
            # 工厂：用于为并集子模型创建新 GaussianModel
            def factory_fn():
                return GaussianModel(scene.dataset.sh_degree, gaussian_dim=gaussians.gaussian_dim,
                                     time_duration=scene.time_duration, rot_4d=gaussians.rot_4d,
                                     force_sh_3d=gaussians.force_sh_3d,
                                     sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
            # 我们对“要渲染的这组 views”预计算关键帧掩码
            # （想更稳，可以用训练集 views 来算；接口是一样的）
            keyframes, active_sets = build_keyframe_active_sets(
                scene.getTestCameras() if not skip_test else scene.getTrainCameras(),
                gaussians, pipeline, background, kf_interval=args.kf_interval, alpha_thresh=args.alpha_thresh
            )
            get_union_model = union_subset_cache_factory(gaussians, opt, factory_fn, active_sets)
```

## D. 在 `render_set()` 的渲染循环里使用子模型

在你当前的 `render_set()` 中，渲染帧的地方是：

```python
rendering = render(view.cuda(), gaussians, pipeline, background)
```

改为（在有过滤时）：

```python
model_for_this_frame = gaussians
if get_union_model is not None and keyframes is not None:
    # 取离当前帧最近的两个关键帧索引
    # 我们用views的顺序来近似时间；若有 view.timestamp 更精确可替换
    cur_idx = idx
    k_ids = [k for k,_ in keyframes]
    # 找到 k_left <= cur_idx <= k_right
    import bisect
    pos = bisect.bisect_left(k_ids, cur_idx)
    left = max(0, pos-1); right = min(len(k_ids)-1, pos)
    model_for_this_frame = get_union_model(left, right)

rendering = render(view.cuda(), model_for_this_frame, pipeline, background)
```

> 至此，“关键帧掩码过滤”已接入：在整段时间里复用**并集子模型**，避免每帧重建子模型，从而把真正送进 raster 的高斯量锐减，FPS 会显著上去。改动都在 `render.py` 内，**无需动训练循环**。

---

# 什么时候需要动 `train1.py`？

* **一般不需要。**1000+ FPS 的主要收益来自**渲染时的筛选**（关键帧掩码过滤），上面我们已经在 `render.py` 里实现了。
* **如果你还想做“全局裁剪后的微调”**（论文里用来把 PSNR 再抻回来一点），可以在 `train1.py` 最后（或单独脚本）加一个**短程 fine-tune**：

  1. 用上面 `--use_global_prune` 得到的**精简 checkpoint**；
  2. 载入后**关闭 densify/split/clone**，只优化现有高斯（SH/外观/少量位置）；
  3. 跑个 **5k–10k iter** 就够。你的 `train1.py` 本身已经有保存/恢复与评测的完整骨架，**直接复用**即可，不需要结构性重写。

---

# 启动方式（建议组合）

* 只开关键帧过滤：

  ```bash
  python render.py --config your_cfg.yaml --iteration best --use_kf_filter --kf_interval 20 --alpha_thresh 0.01
  ```
* 叠加一次性剪枝（更快/更小）：

  ```bash
  python render.py --config your_cfg.yaml --iteration best \
    --use_global_prune --prune_ratio 0.80 \
    --use_kf_filter --kf_interval 20 --alpha_thresh 0.01
  ```
* （可选）对**剪枝后的 checkpoint**做短程微调：

  ```bash
  python train1.py --config your_cfg.yaml --start_checkpoint output/.../chkpnt_best_pruned.pth \
    --iterations 5000 --densify_until_iter 0   # 关闭 densify
  ```

  （参数名以你工程实际为准，思路是“只调现有点，不再增删点”）

---

## 小贴士与坑位

* **掩码阈值**：`alpha_thresh=0.01~0.02` 常用；过大容易漏，过小过滤收益下降。
* **关键帧间隔**：默认 20；剪枝后可以适当上调（激活集更稳定）。
* **缓存**：我们按“相邻关键帧对”缓存 union 子模型，避免每帧建模开销。
* **统计开销**：全局裁剪的打分统计可以抽样 50 帧左右就够排序，速度很快。
* **质量守恒**：若你对 PSNR/SSIM 很敏感，推荐在剪枝后做 5k 左右微调。

---

**一句话结论**：
想拿到 1000+ FPS 的绝大部分收益，**改 `render.py`（关键帧掩码过滤）即可**；“全局裁剪”可作为**一次性后处理**同样放在 `render.py`；只有当你想进一步把质量抻满，才需要在 `train1.py` 里跑一个**很短的微调回合**。&#x20;

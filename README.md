## 🛠️ Pipeline
<div align="center">
  <img src="./stendogs1.png"/>
</div><br/>


## Get started

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/fudan-zvg/4d-gaussian-splatting
cd DyGS
conda env create --file environment.yml
conda activate 4dgs
```

### Data preparation

**DyNeRF dataset:**

看微信群里的压缩包，里面是预处理好的数据集，后续运行需要更改./configs/endoNerf/cutting.yaml内数据集的路径

## TODO

1. 增加loss，进行多方面的supervision
2. 加入mask，计算除器材遮挡部分的loss
3. (alternative)不用mask遮盖，在数据训练前，先用inpainting method先把图像被手术器材遮挡的部分先补全，随后进行后续的训练
4. 最好给出自己的contributions，例如在loss上下手，或者再gaussian模型上下手，目前倾向于前者。
5. paper







- 预处理（基础数据准备）
```bash
python scripts/pre_dam_dep.py --dataset_root data/endonerf/pulling_soft_tissues --rgb_paths images
```

```

- 训练（示例：pulling 配置）
```bash
python train1.py --config configs/endoNerf/pulling.yaml
```

python train1.py ---config configs/endoNerf/pulling.yaml --use_scale_depth --lambda_si 0.3 --lambda_depth_grad 0.03 --key_every 30 --key_min_gap 10 --key_boost_enac 2.0 --key_boost_depth 2.0 --key_boost_rgb 1.0


0911
python3 /root/autodl-tmp/ST-Endo4DGS-main/train1.py   --config /root/autodl-tmp/ST-Endo4DGS-main/configs/endoNerf/pulling.yaml   --iterations 7000 --eval_interval 500

基础上训练
python3 /root/autodl-tmp/ST-Endo4DGS-main/train.py   --config /root/autodl-tmp/ST-Endo4DGS-main/configs/endoNerf/pulling.yaml   --start_checkpoint /root/autodl-tmp/ST-Endo4DGS-main/output/endonerf/pulling/chkpnt_best.pth

- 渲染（使用 best checkpoint，跳过训练集可视化与视频导出）
```bash
python render.py --config configs/endoNerf/pulling.yaml \
  --checkpoint output/endonerf/pulling/chkpnt_best.pth \
  --skip_train --skip_video --measure_raster_only
```
单独测 FPS（不影响评测质量）
python /root/autodl-tmp/ST-Endo4DGS-main/render.py \
  --config /root/autodl-tmp/ST-Endo4DGS-main/configs/endoNerf/pulling.yaml \
  --iteration best \
  --checkpoint /root/autodl-tmp/ST-Endo4DGS-main/output/endonerf/pulling/chkpnt_best.pth \
  --skip_train --skip_video \
  --measure_raster_only

先恢复满质量评测（不做任何裁剪/筛点）
python /root/autodl-tmp/ST-Endo4DGS-main/render.py \
  --config /root/autodl-tmp/ST-Endo4DGS-main/configs/endoNerf/pulling.yaml \
  --iteration best \
  --checkpoint /root/autodl-tmp/ST-Endo4DGS-main/output/endonerf/pulling/chkpnt_best.pth \
  --skip_train --skip_video


- 评估（计算指标并汇总）
```bash
python metrics.py -m output/endonerf/pulling
```







python3 /root/autodl-tmp/ST-Endo4DGS-main/train1.py   --config /root/autodl-tmp/ST-Endo4DGS-main/configs/endoNerf/pulling.yaml   --iterations 7000 --eval_interval 500


python render.py --config configs/endoNerf/cutting.yaml   --checkpoint output/endonerf/cutting/chkpnt_best.pth   --skip_train --skip_video --measure_raster_only
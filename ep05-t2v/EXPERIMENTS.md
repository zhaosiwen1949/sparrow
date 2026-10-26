# 第 5 集 · 补充对照

## 两种视频表示

同一张主干结构卡片，分别接 Wan 时空 VAE 与逐帧 FLUX.2 图像 VAE；都从零只练动画 4000 步。两种 VAE、潜变量维度、投影参数和序列长度都不同，不能把这个实验说成只拨掉时间压缩。

下面承接 README 的环境、数据、判卷器和变量，在 `code/` 执行。按顺序运行即可观察质量；原实测的耗时比例来自同组并行运行，本机重跑的绝对耗时会随硬件与负载变化。

```bash
python encode.py --data "$W/data" --judges "$W/judges" \
  --out "$W/cache" --parts vidimg
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/representation_wan" --tasks t2v --steps 4000 \
  --seed 0 --rng paired --maxlen 1040 --vid_lat vid
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/representation_framevae" --tasks t2v --steps 4000 \
  --seed 0 --rng paired --maxlen 1040 --vid_lat vidimg
```

| 实测 | Wan 时空 VAE | 逐帧 FLUX.2 VAE |
|---|---:|---:|
| 视频段向量数 | 320 | 1024 |
| 完整序列长度 | 325 | 1029 |
| 视频入口投影参数 | 6528 | 12672 |
| 模型参数量（M） | 16.29 | 16.34 |
| 训练耗时（秒） | 265.6 | 1120.5 |
| 每步耗时（秒，训练耗时 / 4000） | 0.0664 | 0.280125 |
| 训练张量显存峰值（GB） | 5.34 | 16.02 |
| 视频分 | 100%（192/192） | 100%（192/192） |

逐帧版每步慢 4.22 倍，显存峰值为 3 倍；视频分没有区分度。逐帧版的显存需求明显高于主线，不能套用主线的 7.06 GB。完整原值在 `reports/representation_wan.json` 和 `reports/representation_framevae.json`。本目录提供这组报告和训练入口，不附这两份参数。

## 相邻帧变化

`reports/attention_full.json` 与 `reports/attention_frame.json` 的 `eval.t2v.adjacent_rmse_by_action` 按动作记录 15 对相邻帧的每像素通道均方根误差；`judge_check.real` 和 `judge_check.roundtrip` 给真片与 VAE 往返片的参照。

“向左跑”的时间槽边界（第 0→1、4→5、8→9、12→13 帧）：

| 版本 | 0→1 | 4→5 | 8→9 | 12→13 |
|---|---:|---:|---:|---:|
| 真片 | 25.581 | 25.343 | 25.627 | 25.635 |
| 互看 | 26.569 | 26.480 | 27.306 | 27.740 |
| 只看自己 | 40.176 | 37.748 | 42.001 | 41.113 |

以 JSON 原值为准；这些值描述本次样例的变化，不能把像素变化小直接等同于动画更好。


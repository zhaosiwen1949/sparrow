---
license: apache-2.0
language:
- zh
tags:
- text-to-image
- flow-matching
- classifier-free-guidance
- from-scratch
- education
- toy-model
---

# 麻雀 · 第 2 集 · 文字→图(从零训练 4000 轮)

《从零训一个多模态模型》第 2 集训好的权重。约 16M 参数的小 Transformer,只练「文字→图」:输入颜色与动物两个词(如「红色 小猫」),在冻结的 FLUX.2 VAE 潜空间里用流匹配生成 32×8×8 的格子,还原成 64×64 的卡通小动物。玩具规模,用来看懂原理,不是商用模型;没有基础模型,从零训练。

- 词表:颜色 红色 橙色 黄色 绿色 蓝色 紫色 粉色 棕色;动物 小猫 小狗 小兔 小鸟 小鱼 小熊。
- 训练:主干 8 层、宽 384、6 头;每轮一批 512 张,AdamW 6e-4,4000 轮,EMA 0.999;约一成题把两个词换成「不给题」,画图时用无条件引导(引导强度 3,30 步)。
- 训练数据:程序生成的卡通小世界练习图 6 万张(测试图 2000 张,种子不重叠)。

## 环境与硬件

- 一块 NVIDIA 显卡。训练显存峰值约 5 GB(`report.json` 的 `train_peak_mem_GB` 为 4.51),评测约 2.5 GB;只做推理更少。
- 容器 `nvcr.io/nvidia/pytorch:25.09-py3`(PyTorch 2.9),另装 `diffusers==0.40.0 transformers==5.17.0 accelerate==1.15.0 huggingface_hub==1.32.0 soundfile==0.13.1`。
- 冻结零件:FLUX.2 的图像 VAE(`hf download black-forest-labs/FLUX.2-klein-4B --include "vae/*"`,约 0.16 GB,Apache-2.0)。

## 用法

代码在本集目录的 `code/`:

```bash
cd ../code
python infer.py ../weights/final.pt t2i "红色 小猫" cat.png     # 30 步、引导强度 3,输出 256×256 PNG
```

在这份权重上接着训:`python sparrow.py ... --init ../weights/final.pt --tasks ...`。

自己从零训出这份权重:造数据 → 判卷模型 → 编码 → 训练 4000 轮 → 画图 → 判卷,完整命令见本集 [README](../README.md) 第 3 节。核心一条:

```bash
python sparrow.py --data $W/data --cache $W/cache --judges $W/judges --out $W/runs/ep02 --tasks t2i --steps 4000 --eval_every 250
```

## 文件

| 文件 | 是什么 | sha256 |
|---|---|---|
| `final.pt` | 片中用的那份权重:按本集训练命令(没带 `--eval_every`)从零训练到第 4000 轮(训练结束)的 EMA 权重;内含 `model`(参数)、`args`(训练参数)、`stat`(答案格子的归一化统计量) | `c13f94c0a07afdfb85143380be9716b012026f74ce01f7fa2d8a4e3f87f55d29` |
| `report.json` | 训练参数、每 100 轮的 loss、终评 | 见上一级目录 `MANIFEST.sha256` |

## 评测

画对率:生成图交给只用真图训的小 CNN 判动物和颜色,两样都对才算对。

- 这份权重终评:384 题(6 动物 × 8 颜色 × 8 张)画对率 100.0%,动物、颜色各 100%。
- loss:第 0 轮 1.9978、第 1000 轮 0.4561、第 3900 轮 0.2918;同一条命令再从零训一次,走势一致(第 1000 轮 0.4534、第 3900 轮 0.2921),终评同样 384/384。
- 片中「步数」「引导强度」两组对照就是用这份权重测的。
- 片中对照实验(步数、引导强度、训练进度)的实测表见本集 [README](../README.md) 第 4 节。

## 局限

- 只会 6 种动物 × 8 种颜色的组合,画面 64×64,风格只有程序画的卡通小世界。
- 画出来的会不会只是复制练习图、没见过的组合画不画得出,本集没有测。

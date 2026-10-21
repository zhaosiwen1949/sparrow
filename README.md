---
license: apache-2.0
language:
- zh
tags:
- multimodal
- text-to-image
- flow-matching
- from-scratch
- education
- toy-model
---

# 麻雀:从零训一个多模态模型 · 第 2 集(生图)

视频课程《从零训一个多模态模型》第 2 集(生图)的配套代码与训好的权重,内容在 [`ep02-t2i/`](ep02-t2i/)。这一集从零训练一只约 16M 参数、只会「照字画图」的小模型「麻雀」:训练时学从噪声指向答案的方向,画图时从一团噪声出发,一步步顺着方向走,并用文字引导画出「红色 小猫」这样的 64×64 图。

第 4 集（生声音）的配套代码、权重、训练录音与样例见 [`ep04-t2a/`](ep04-t2a/)。
模型仓库:https://modelscope.cn/models/llmzerotohero/sparrow

## 硬件

一块 NVIDIA 显卡,训练显存峰值约 5 GB;要下载的只有 FLUX.2 的图像 VAE,约 0.16 GB。

## 引用

视频:B 站「大模型成长之路」合集《从零训一个多模态模型 | 把技术讲清楚》第 2 集:https://www.bilibili.com/video/BV1zDpQ6zE8n

```bibtex
@misc{sparrow2026,
  title        = {从零训一个多模态模型 · 第 2 集 生图},
  author       = {大模型成长之路},
  year         = {2026},
  howpublished = {视频课程,B 站「大模型成长之路」,https://www.bilibili.com/video/BV1zDpQ6zE8n}
}
```

## 许可证

代码与权重见 [`LICENSE`](LICENSE)。训练数据全部由程序生成。用到的冻结零件只有 FLUX.2 的图像 VAE,许可证为 Apache-2.0。

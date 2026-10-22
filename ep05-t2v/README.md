# 第 5 集 · 生视频：先压时间，帧与帧互看，旧题照练

《从零训一个多模态模型》第 5 集的代码、训练命令、参数与样例。接着第 4 集的权重，让同一个约 16.04M 参数的主干学演「红色 小猫 向左跑」：冻结的 Wan 时空 VAE 把 16 帧动画压成 5 个时间槽，主干用注意力让这些时间槽互看，再用流匹配从噪声走出动画；画图、念名字两种旧题一起练。本集是 64×64、16 帧的教学玩具模型。

![主线实测：红色小猫向左跑](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/course_ep5/gen_t2v_0_红色小猫向左跑.gif)

## 1. 环境与硬件

使用 Python 3.12、支持 CUDA 的 NVIDIA 显卡。实测软件环境为 `nvcr.io/nvidia/pytorch:25.09-py3`（PyTorch 2.9）；下列安装命令在已有该 PyTorch 环境的前提下执行。主线用单张 48 GB 显卡：4000 步训练 **1074.2 秒（约 17.9 分钟）**，终评另用 **41.7 秒**；训练张量显存峰值 **7.06 GB**，终评 **5.56 GB**，来自 `weights/course_ep5/report.json`。这是已编码数据上的训练时间，不含下载、造数据、训练判卷器和编码；未测本集最低可用显存、主机内存或完整准备流程总耗时。

片中资源卡的“约 12 分钟、7.1 GB”来自训练进度那次运行：训练 738.9 秒、张量显存峰值 7.13 GB（`reports/progress/final.json`）。它采用 `paired` 随机协议，并每 250 步评一次；与这里发布的主线权重是不同运行，两个耗时不能混用。

本集只需图像 VAE、Mimi 和新增的 Wan 视频 VAE。首次声音评测还会下载 Paraformer-zh；直接生成动画只需 Wan VAE。下面从本集目录开始：

```bash
cd code
pip install -r requirements.txt
hf download black-forest-labs/FLUX.2-klein-4B --include 'vae/*'
hf download kyutai/mimi --include '*.json' --include '*.safetensors' --include '*.txt'
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include 'vae/*'
```

显存数是 PyTorch 的最大已分配张量字节数除以 10⁹，未包含全部 CUDA 上下文与其他进程占用，不能当作显卡容量下限。

## 2. 照着训练

以下继续在 `code/` 执行。`W` 是自己的工作目录。通过公开下载地址取得第 4 集主线参数和已把关录音，避免把 Git LFS 指针当成权重；也可把 `INIT` 指向自己训好的第 4 集参数。

```bash
W=../work
INIT="$W/init/course_ep4.pt"
BASE=https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master
mkdir -p "$W/init" "$W/data" "$W/judges" "$W/runs"

curl -fL "$BASE/ep04-t2a/weights/course_ep4/final.pt" -o "$INIT"
curl -fL "$BASE/ep04-t2a/data/voice.npz" -o "$W/data/voice.npz"
printf '%s  %s\n' \
  2b7dfe25ed142c041bebba02ea112b22cfaf97155917bf89a0c3ce5b647d8d8c "$INIT" \
  e3fb4fcb76cd456db9bfd38aa3ad4287c22a913a085f07a90b0f7be33297a634 "$W/data/voice.npz" \
  | sha256sum -c -

# 造图和动画：训练、测试使用不同种子区间
python data.py world "$W/data/world_test.npz" 2000 500 900000000
python data.py world "$W/data/world_train.npz" 60000 10000 0
python train_judges.py img --data "$W/data" --out "$W/judges" --steps 2000
python train_judges.py vid --data "$W/data" --out "$W/judges" --steps 1500
python encode.py --data "$W/data" --judges "$W/judges" \
  --out "$W/cache" --parts img,aud,vid

# 接第 4 集，画图、念名字、演动画一起练
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/course_ep5" --init "$INIT" --tasks t2i,t2a,t2v \
  --steps 4000 --seed 0 --rng legacy --vid_attn full --eval_every 0
```

每步画图 512 条、念名字 512 条、动画 128 段；损失权重分别为 1、4、1。图像和声音的归一化统计量沿用第 4 集，新增视频统计量从训练集计算。训练输出 `final.pt`、`report.json` 和 `samples/`；`--out` 必须是没有运行产物的新目录。

代码与当前实验入口一致，发布参数来自主线实测；当前评测按题号固定起点噪声，原主线终评采用另一套抽噪声顺序，重跑不承诺逐字节一致。发布的第 4 集参数与实验实际起点的全部 132 个模型及统计量张量已核验一致。

## 3. 用本集参数生成动画

在 `code/` 执行；不需要训练数据、判卷器或编码缓存：

```bash
W=../work
python infer.py ../weights/course_ep5/final.pt \
  t2v '红色 小猫 向左跑' "$W/cat-left.gif"
python infer.py ../weights/course_ep5/final.pt \
  t2v '橙色 小狗 向右跑' "$W/dog-right.gif"
python infer.py ../weights/course_ep5_frame/final.pt \
  t2v '红色 小猫 向左跑' "$W/cat-frame.gif"
```

默认生成 30 步、CFG 3、噪声种子 0，可用 `--steps`、`--cfg`、`--seed` 更改。入口还保存起点噪声与 16 张原始帧。颜色为红色、橙色、黄色、绿色、蓝色、紫色、粉色、棕色；动物为小猫、小狗、小兔、小鸟、小鱼、小熊；动作为向左跑、向右跑、跳、转身。

所附 GIF 是实测输出，16 帧、每帧 100 ms，64×64 结果按最近邻放大为 192×192。当前直接推理入口显示为 256×256，主线原终评抽噪声顺序也不同，因此同一句提示不保证得到与所附 GIF 逐字节相同的文件。

| 参数文件 | 大小 |
|---|---:|
| `weights/course_ep5/final.pt` | 64,203,914 字节（约 61.23 MiB） |
| `weights/course_ep5_frame/final.pt` | 64,203,921 字节（约 61.23 MiB） |

![主线实测：橙色小狗向右跑](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/course_ep5/gen_t2v_1_橙色小狗向右跑.gif)

## 4. 可选：主干里的“只看自己”

准备步骤相同，从同一份第 4 集参数另训一份：

```bash
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/course_ep5_frame" --init "$INIT" --tasks t2i,t2a,t2v \
  --steps 4000 --seed 0 --rng legacy --vid_attn frame --eval_every 0

# 用统一的评测噪声分别评两份已发布参数
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/attention_full" --eval_only \
  --ckpt ../weights/course_ep5/final.pt --eval_tasks t2v --eval_seed 0
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/attention_frame" --eval_only \
  --ckpt ../weights/course_ep5_frame/final.pt --eval_tasks t2v --eval_seed 0
```

这里的“帧”是压缩后的时间槽：16 帧末尾复制一帧成为 17 帧，Wan VAE 编为 `[16, 5, 8, 8]`（通道、时间、高、宽）；每槽 64 个向量，共 320 个，加文字后序列长 325。`full` 允许视频段内互看；`frame` 只允许同槽内互看及读取前面的文字。两版的 Wan VAE 本身都跨时间处理，所以不能说整个模型完全没有跨帧信息。

| 测量 | 互看 | 只看自己 | 出处 |
|---|---:|---:|---|
| 片中采用的统一口径重评 | 99.5%（191/192） | 97.4%（187/192） | `reports/attention_*.json` |
| 原训练终评 | 99.5%（191/192） | 95.8%（184/192） | `weights/*/report.json` |

视频分由独立视频 CNN 判动物、颜色、动作，三项都对才算对。新口径的差别在动作；两个口径不能混用。主线原终评还记录画图 100%（384/384）、念名字 97.9%（235/240）。逐帧图像判卷器在真动画上仅 90.9%，未过 97% 校准线，报告中的逐帧一致率标为无效，不拿它比较优劣。

同题同噪声重评样例：

![互看版](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/attention_full/gen_t2v_0_红色小猫向左跑.gif)
![只看自己版](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/attention_frame/gen_t2v_0_红色小猫向左跑.gif)

## 5. 训练进度与核验

片中每 250 步评一次的曲线是另一次 `paired` 随机协议运行，命令为：

```bash
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/progress" --init "$INIT" --tasks t2i,t2a,t2v \
  --steps 4000 --seed 0 --rng paired --eval_every 250 --eval_tasks t2v
```

这次视频分为 250 步 1.0%（2/192）、750 步 67.2%（129/192）、4000 步 100%（192/192）。按未四舍五入的准确率，首次达到 99% 的是动作 1000 步、动物 1500 步、颜色 1750 步；“首次达到”不表示以后不回落。报告在 `reports/progress/`，三段实测动画在 `samples/progress/`；本目录未附这次曲线运行的参数，发布主线的 99.5% 与曲线末点 100% 属于不同运行。

![250 步](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/progress/step250.gif)
![750 步](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/progress/step750.gif)
![4000 步](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep05-t2v/samples/progress/step4000.gif)

两种视频表示的补充对照见 [EXPERIMENTS.md](EXPERIMENTS.md)。`PROVENANCE.json` 记录来源与发布哈希；权重仅替换路径元数据，全部模型与归一化统计量张量的存储字节及描述信息不变。报告保留实测指标、loss 日志和原权重哈希，原哈希与发布哈希的对应关系见该文件。在本集目录核验：

```bash
sha256sum -c MANIFEST.sha256
```

## 6. 许可与署名

代码与麻雀参数采用 [Apache-2.0](LICENSE)。图像、动画由 `code/world/world.py` 生成。外部冻结零件单独下载，遵循各自许可证：

- Wan2.1 VAE — Wan-AI：[模型](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)，Apache-2.0。
- FLUX.2 图像 VAE — Black Forest Labs：[模型](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)，Apache-2.0。
- Mimi — Kyutai：[模型](https://huggingface.co/kyutai/mimi)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。
- Paraformer-zh — FunASR：[模型](https://huggingface.co/funasr/paraformer-zh)，Apache-2.0。
- 训练录音沿用[第 4 集数据与署名](../ep04-t2a/README.md)：432 条合成录音，324 条训练、108 条测试；FLEURS — Google 的普通话参考音及衍生内容按 CC-BY-4.0 署名，CosyVoice — FunAudioLLM 用于合成。完整处理与许可说明见第 4 集。本目录通过下载命令复用该录音包。


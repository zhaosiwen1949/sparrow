# 第 6 集 · 看图听声：把输入变成标签，练会文字出口

《从零训一个多模态模型》第 6 集的代码、训练命令、参数与样例。接着第 5 集的麻雀练 4000 轮，新增看图、听声、看视频说标签三种题，画图、念名字、演动画三种旧题照练。冻结的 SigLIP 2 和 Whisper 编码器先把输入变成数字，再经可训练的输入接口送进同一个约 16.04M 参数的主干；LM head 为固定词表中的全部 27 项打分。训练时，出题程序保存的颜色、动物、动作组成标准答案，loss 推动正确项相对其他项的分数提高；回答时逐格选最高项，把已经选出的标签接回前文，继续回答。这里是 64×64 卡通世界的教学玩具模型，回答的是固定标签。

![主线实测输入：看图说“红色 小猫”](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/lesson/frame0.png)

## 1. 环境与实测资源

使用 Python 3.12、支持 CUDA 的 NVIDIA 显卡。实测软件环境为 `nvcr.io/nvidia/pytorch:25.09-py3` 的 PyTorch 2.9；以下安装命令在已装好这一 PyTorch 环境的前提下执行。主线在单张 **48 GB 显卡**上完成：

| 阶段 | 实测 |
|---|---:|
| 已编码数据上的 4000 轮训练 | 1474.8 秒，约 24.6 分钟 |
| 训练张量显存峰值 | 7.07 GB |
| 终评耗时 | 43.6 秒 |
| 终评张量显存峰值 | 5.56 GB |

来源为 `weights/course_ep6/report.json`。显存采用 PyTorch 最大已分配张量字节数除以 10⁹，不包括全部 CUDA 上下文和其他进程占用；未测最低显卡容量、主机内存需求或完整准备流程总耗时。训练时间不含下载、造数据、训练判卷器和冻结零件编码。这些是本次运行的资源记录。

下面从本集目录开始。生成三种旧题仍需要图像 VAE、Mimi 和视频 VAE；本集新增理解用的 SigLIP 2、Whisper tiny。声音生成评测另会下载 Paraformer-zh。

```bash
cd code
pip install -r requirements.txt
hf download black-forest-labs/FLUX.2-klein-4B --include 'vae/*'
hf download kyutai/mimi --include '*.json' --include '*.safetensors' --include '*.txt'
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include 'vae/*'
hf download google/siglip2-base-patch16-224 --include '*.json' --include '*.safetensors'
hf download openai/whisper-tiny --include '*.json' --include '*.safetensors' --include '*.txt'
```

只用发布权重看图、听声、看视频时，分别需要 SigLIP 2、Whisper、SigLIP 2，无需先训练或准备生成用的 VAE、Mimi、判卷器。

## 2. 接着第 5 集训练

以下命令均在 `code/` 执行。`W` 是自己的工作目录，`INIT` 可以换成自己训练的第 5 集参数。公开下载地址直接取得实体文件，随后检查 SHA-256，避免误用 Git LFS 指针。

```bash
W=../work
INIT="$W/init/course_ep5.pt"
BASE=https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master
mkdir -p "$W/init" "$W/data" "$W/judges" "$W/runs"
curl -fL "$BASE/ep05-t2v/weights/course_ep5/final.pt" -o "$INIT"
curl -fL "$BASE/ep04-t2a/data/voice.npz" -o "$W/data/voice.npz"
printf '%s  %s\n' \
  2fdfa161614d2703cfc4b353b047df62bbcdd9d9c65f07cf233013123d41cbae "$INIT" \
  e3fb4fcb76cd456db9bfd38aa3ad4287c22a913a085f07a90b0f7be33297a634 "$W/data/voice.npz" \
  | sha256sum -c -

# 训练、测试采用不同种子区间，图与动画由程序生成
python data.py world "$W/data/world_test.npz" 2000 500 900000000
python data.py world "$W/data/world_train.npz" 60000 10000 0
python train_judges.py img --data "$W/data" --out "$W/judges" --steps 2000
python train_judges.py vid --data "$W/data" --out "$W/judges" --steps 1500
python encode.py --data "$W/data" --judges "$W/judges" \
  --out "$W/cache" --parts img,aud,vid,see,hear,watch

# 三种新题与三种旧题共同训练
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/course_ep6" --init "$INIT" \
  --tasks t2i,t2a,t2v,i2t,a2t,v2t --steps 4000 \
  --seed 0 --rng legacy --vid_attn full --eval_every 0
```

若第 5 集的同一份数据、判卷器和缓存已经齐全，可以复用它们，跳过造数据及判卷器步骤，仅补 `encode.py --parts see,hear,watch` 的理解特征；其余参数与上面的编码命令相同。

每轮六种题都参与：画图、念名字、看图、听声各 512 条，演动画、看视频各 128 段。念名字的 loss 权重为 4，其他任务为 1。沿用第 5 集的生成特征归一化统计量，新增理解特征统计量从训练数据计算。输出目录必须没有已有运行产物；完成后写入 `final.pt`、`report.json` 和 `samples/`。

已核验公开第 5 集参数与本次训练实际起点的全部 128 个模型张量、6 个统计量张量完全一致。所附通用代码为当前实验入口；生成任务终评按题号固定噪声，而所附原报告使用全局随机流，所以重新评测生成题不保证得到相同分数或逐字节相同的样例。理解题逐格贪心回答，不使用生成噪声。

## 3. 直接看图、听声、看视频

继续在 `code/` 执行，输入文件已经附在本目录中。无需数据缓存和判卷器：

```bash
python infer.py ../weights/course_ep6/final.pt i2t ../samples/lesson/frame0.png
python infer.py ../weights/course_ep6/final.pt a2t ../samples/lesson/voice_cat.wav
python infer.py ../weights/course_ep6/final.pt v2t ../samples/lesson/anim.gif
```

这三个样例的真实输出记录在 `samples/lesson/answers.json`：

| 输入 | 实测输出 |
|---|---|
| `frame0.png`：下面动画的第 0 帧 | `红色 小猫 <eos>` |
| [voice_cat.wav](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/lesson/voice_cat.wav)：合成录音“小猫” | `小猫 <eos>` |
| `anim.gif`：第 5 集麻雀生成的红猫向左跑动画 | `红色 小猫 向左跑 <eos>` |

![同一主线权重看视频的实测输入](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/lesson/anim.gif)

`<eos>` 在片中叫“说完”。图像回答三格、声音两格、视频四格，循环长度由题型固定，不靠遇到结束标签提前停止。每一格都对全部 27 项打分，没有先按颜色、动物、动作筛候选。训练时，标准答案的前文参与计算，各目标格只读前文；回答时，前文则是自己已经选出的标签。

`i2t` 会把输入图缩到 64×64；`a2t` 要求录音去静音后，加上 50 毫秒前导仍能装进 2.08 秒；`v2t` 接收 16 帧 GIF、16 帧数组 `.npy` 或恰含 16 张 PNG 的目录。本集看视频取全部 16 帧。所附音频为 24 kHz 单声道、2.08 秒。

同一段动画在第 5 集权重上得到的真实错答是 `<vid> 小兔 向右跑 跳`，也保留在 `answers.json`。当时只补用了第 6 集的 `watch` 归一化统计量，模型参数仍是第 5 集；该记录与训练后的回答使用同一输入。

另有原终评直接导出的主线样例，文件名保留模型真实回答：

![测试图：模型说绿色小鸟](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/course_ep6/i2t_0_模型说绿色小鸟.png)

- [测试录音：模型说小兔](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/course_ep6/a2t_0_模型说小兔.wav)
- [测试动画：模型说红色小狗转身](https://modelscope.cn/models/llmzerotohero/sparrow/resolve/master/ep06-understand/samples/course_ep6/v2t_0_模型说红色小狗转身.gif)

`samples/course_ep6/` 还保留同一次终评的画图、念名字、演动画输出。看图、听声、看视频样例是输入；以 `gen_` 开头的是生成输出。

## 4. 主线成绩与文字出口

| 新增任务 | 第 5 集参数不训练直接考 | 接着练 4000 轮后 |
|---|---:|---:|
| 看图说颜色和动物 `i2t` | 0/2000（0%） | 2000/2000（100%） |
| 听声说动物 `a2t` | 0/108（0%） | 108/108（100%） |
| 看视频说颜色、动物和动作 `v2t` | 0/500（0%） | 500/500（100%） |

出处：`reports/before_training.json` 与 `weights/course_ep6/report.json`。按整题检查内容标签，不检查结束标签；0% 不等于每个标签都错，100% 也不等于任意现实输入都能理解。听声的 108 条测试录音来自训练划分里未出现的 3 个音色。

旧题终评为画图 100%（384/384）、念名字 96.7%（232/240）、演动画 99.5%（191/192）。上一集念名字为 97.9%（235/240），本次多错 3 条；画图和视频分与上一集相同。这些是单次运行记录。

词表固定为 8 个颜色、6 个动物、4 个动作和 9 个记号，共 27 项；“向左跑”整体是一项。LM head 是算分的层，词表是可选标签清单，标准答案则是当前题目应该选的标签。训练改变主干、输入接口、LM head 等参数，不会修改标签文字或扩词表；冻结编码器不更新。`reports/train.jsonl` 保留原训练日志的 40 条数值记录，与报告中的 `log` 一致。片中“推高红色、压低其他项”是当前目标的训练方向示意，不能据此断言混合训练的每次更新都让某个原始分数单调变化。

在上面准备好数据、缓存、判卷器后，可单独评测本集权重：

```bash
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/ep6_eval" --eval_only --ckpt ../weights/course_ep6/final.pt \
  --eval_tasks t2i,t2a,t2v,i2t,a2t,v2t --eval_seed 0
```

## 5. 文件与校验

| 文件 | 内容／大小 |
|---|---|
| `code/` | 模型、造数据、冻结编码、判卷器和直接推理入口 |
| `weights/course_ep6/final.pt` | 主线权重，64,220,444 字节（约 61.25 MiB） |
| `weights/course_ep6/report.json` | 主线配置、训练 loss、资源与终评 |
| `reports/before_training.json` | 接第 5 集、零轮训练的理解题原始评测 |
| `samples/` | 主线真实输出、片中输入与逐格回答记录 |
| `PROVENANCE.json` | 来源哈希、公开文件哈希、张量一致性核验 |

权重仅替换 5 个路径元数据字符串，全部 128 个模型张量与 12 个归一化统计量张量的描述和存储字节不变。原权重 SHA-256 与公开文件 SHA-256 的映射见 `PROVENANCE.json`。本目录只附本集主线权重；冻结预训练零件通过上面的命令单独下载。

在本集目录执行：

```bash
sha256sum -c MANIFEST.sha256
```

## 6. 许可与署名

代码与麻雀参数采用 [Apache-2.0](LICENSE)。图像和动画由 `code/world/world.py` 生成。外部零件独立下载，遵循各自许可证：

- SigLIP 2 — Google：[模型与许可](https://huggingface.co/google/siglip2-base-patch16-224)，Apache-2.0。
- Whisper tiny — OpenAI：[下载仓库与许可](https://huggingface.co/openai/whisper-tiny)，该下载仓库标注 Apache-2.0；本集使用其编码器。
- FLUX.2 图像 VAE — Black Forest Labs：[模型](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)，Apache-2.0。
- Wan2.1 视频 VAE — Wan-AI：[模型](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)，Apache-2.0。
- Mimi — Kyutai：[模型](https://huggingface.co/kyutai/mimi)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。
- Paraformer-zh — FunASR：[模型](https://huggingface.co/funasr/paraformer-zh)，Apache-2.0。
- 训练录音及本集所附真实录音样例沿用[第 4 集数据与署名](../ep04-t2a/README.md)：432 条合成录音，324 条训练、108 条测试；Google FLEURS 普通话参考音及其衍生内容按 CC-BY-4.0 署名，FunAudioLLM CosyVoice 用于合成。完整数据处理与许可说明见第 4 集。

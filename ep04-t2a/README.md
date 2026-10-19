# 第 4 集 · 生声音：声音压短照画图走，旧题带着练

《从零训一个多模态模型》第 4 集的代码、训练命令、参数与样例。上一集，麻雀会画红色小猫；这一集让同一个主干学念「小猫」：录音先由冻结的 Mimi 压成连续的声音格子，再照画图的流匹配走法，从噪声一步步生成。主线念名字正确率 **97.5%**，画图仍为 **100%**。这是只念六种动物名的教学玩具模型。

## 1. 环境与安装

使用支持 CUDA 的 NVIDIA 显卡和 Python 3.12。原实验软件环境为 `nvcr.io/nvidia/pytorch:25.09-py3`（PyTorch 2.9）；主线训练峰值 7.1 GB，8 GB 显卡偏紧。下面命令从本集目录执行：

```bash
cd code
pip install -r requirements.txt
hf download black-forest-labs/FLUX.2-klein-4B --include 'vae/*'
hf download kyutai/mimi --include '*.json' --include '*.safetensors' --include '*.txt'
```

训练只用图像 VAE 和 Mimi；首次念名字评测时，FunASR 会下载 Paraformer-zh。直接出声只需要 Mimi，直接画图只需要图像 VAE。

## 2. 照着训练

以下命令在 `code/` 执行。`W` 是观众自己的工作目录。下载同仓库第 2 集的 `weights/final.pt` 作为起点；若自己已经训好第 2 集，也可将 `INIT` 换成自己的参数。训练目录须是新目录。

```bash
W=../work
INIT=../../ep02-t2i/weights/final.pt
mkdir -p "$W/data" "$W/judges" "$W/runs"
# 已把关的录音：324 条训练、108 条测试，24 kHz、每条 2.08 秒
cp ../data/voice.npz "$W/data/voice.npz"

# 卡通小世界：测试与训练使用不同的种子区间
python data.py world "$W/data/world_test.npz" 2000 500 900000000
python data.py world "$W/data/world_train.npz" 60000 10000 0
# 编码器启动时会读两个判卷器；本集使用图的那个
python train_judges.py img --data "$W/data" --out "$W/judges" --steps 2000
python train_judges.py vid --data "$W/data" --out "$W/judges" --steps 1500
python encode.py --data "$W/data" --judges "$W/judges" --out "$W/cache" --parts img,aud

# 接着第 2 集练，画图题与念名字题每轮各 512 条
python sparrow.py --data "$W/data" --cache "$W/cache" --judges "$W/judges" \
  --out "$W/runs/course_ep4" --init "$INIT" --tasks t2i,t2a \
  --steps 8000 --seed 0 --rng legacy --eval_every 1000
```

产物为 `final.pt`、`report.json`、`samples/`；每 1000 轮另有中途评测。录音训练时会按固定随机数做四份时间平移增强，只挪入尾部静音，不截字。念名字损失权重为 4，画图为 1。

只下载本集也能先用 `--tasks t2i --steps 4000 --seed 0 --rng legacy` 从零训第 2 集：沿用上述公共参数，把输出设为 `"$W/runs/course_ep2"`，不传 `--init`，再把 `INIT` 指向它的 `final.pt`。

## 3. 用本集参数出样例

仍在 `code/` 执行，不需要训练数据、判卷器或编码缓存：

```bash
python infer.py ../weights/course_ep4/final.pt t2a 小猫 ../work/cat.wav
python infer.py ../weights/course_ep4/final.pt t2a 小狗 ../work/dog.wav
python infer.py ../weights/course_ep4/final.pt t2i '红色 小猫' ../work/cat.png
# 对照模型用相同入口，只替换参数路径
python infer.py ../weights/replay_0/final.pt t2a 小兔 ../work/only-rabbit.wav
python infer.py ../weights/replay_512/final.pt t2i '蓝色 小狗' ../work/replay-dog.png
python infer.py ../weights/course_ep4_only_t2a/final.pt t2a 小鸟 ../work/early-bird.wav
```

默认生成 30 步、引导强度 3、噪声种子 0；可加 `--seed 1` 换起点。六种名字为小猫、小狗、小兔、小鸟、小鱼、小熊。`samples/` 按模型分目录，每份参数有六条音频和两张图，均为该次训练原有的终评样例；主线原终评的噪声抽法与当前推理入口不同，单条输出不承诺逐字节一致。

## 4. 实测与对照

训练记录(report.json)来自较早版本的同一份代码,参数与张量和这里发布的一致;按命令重跑结果相近但不保证逐字节相同。

全部数字为片中单次训练（训练种子 0）的原值。念名字按 Paraformer-zh 听写的拼音判，不计声调；画图由独立 CNN 判动物与颜色，两项都对才算对。

| 模型参数目录 | 画图正确率（384 题） | 念名字正确率（240 条） | 对应实验 |
|---|---:|---:|---|
| `course_ep4` | 100.0%（384/384） | 97.5%（234/240） | 本集主线、4-4 |
| `replay_0` | 0.8%（3/384） | 97.9%（235/240） | 4-2、4-3：只练念名字 |
| `replay_512` | 100.0%（384/384） | 98.3%（236/240） | 4-2、4-3：两样一起练 |
| `course_ep4_only_t2a` | 2.3%（9/384） | 98.3%（236/240） | 片中提过的早先另一次只练声音运行 |

主线和两只新对照是不同运行，97.5% 与 98.3% 不相减。4-1 另从零比较两种声音表示：Mimi 潜变量版 97.9%（235/240），直接波形版 0.0%（0/240）。4-3 每轮只带 10 道画图题，这次也保住了画图：100.0%（384/384）；各档念名字在 95% 到 99% 之间，没看出差别。主线念名字从 1000 轮的 54.6%（131/240）、2000 轮的 89.2%（214/240），到 8000 轮的 97.5%（234/240）。真录音参照 100.0%（108/108），与生成的 240 条是两组题。

完整七档数字与对照命令见 [EXPERIMENTS.md](EXPERIMENTS.md)。本目录提供上表四份参数；4-1 与七档其余参数可按对照命令训练。`weights/*/report.json` 保留实测值，路径改为通用相对路径；参数内全部模型张量和归一化统计量逐元素不变，来源与发布 SHA-256 见 [PROVENANCE.json](PROVENANCE.json)。运行 `sha256sum -c MANIFEST.sha256` 核对本集文件。

## 5. 数据、许可与署名

代码与麻雀参数采用 [Apache-2.0](LICENSE)。图像由 `code/world/world.py` 生成。`data/voice.npz` 是通过把关的 432 条合成录音，保留原数组与训练／测试划分：9 个训练音色、3 个测试音色，测试说话人未用于训练。录音及其 FLEURS 衍生内容按 CC-BY-4.0 署名；代码和麻雀参数的 Apache-2.0 不替代外部零件与数据自身的许可。

- **Mimi — Kyutai**：[模型与来源](https://huggingface.co/kyutai/mimi)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。用作冻结的声音编解码器；麻雀取跳过码本查表的连续潜变量，再由 Mimi 解码。未在本目录重新分发 Mimi 参数。
- **FLEURS — Google**：[普通话数据来源](https://huggingface.co/datasets/google/fleurs)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。使用其中真人朗读作为音色参考，由 CosyVoice 合成动物名；随后去静音、补至 2.08 秒并做识别把关。本目录分发合成后的录音，不含原始朗读。
- **CosyVoice2-0.5B、CosyVoice-300M-SFT — FunAudioLLM / 通义**：[项目](https://github.com/FunAudioLLM/CosyVoice)，[CosyVoice2-0.5B 模型卡](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)、[CosyVoice-300M-SFT 模型卡](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT)均标注 Apache-2.0。用于造录音与候选音色；把关后保留的 12 个音色来自 FLEURS 参考音。造录音的原型脚本在 `code/voice/`，训练直接读取所附录音包。
- **Whisper-large-v3-turbo — OpenAI**：[来源](https://huggingface.co/openai/whisper-large-v3-turbo)，MIT；合成录音时当筛子。
- **Paraformer-zh — FunASR**：[来源](https://huggingface.co/funasr/paraformer-zh)，Apache-2.0；用于最终把关和念名字判卷。
- **FLUX.2 图像 VAE — Black Forest Labs**：[来源](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/main/vae)，Apache-2.0；用于图像编码与还原。

# 05 · 表示方案对照:`ep1_assets.py`

[`ep1_assets.py`](../code/ep1_assets.py) 是第 4 集新增的脚本。它回答一个问题:**为什么非要先用 Mimi 把声音压成潜变量,不能直接生成波形?**

做法是用同一个主干、同样的超参,只换「生成用的表示」,从零只练这一项本事。所以它产出的不是主线权重,而是一组对照数字。

> 第 2 集目录里的 `ep3_assets.py` 是另一个脚本:它用第 2 集的权重做素材(一步步生成、扫采样步数和 CFG、最近邻、VAE 往返),不涉及训练。`ep1_assets.py` 不是它的改版。

## 1. 四种组合

```
python ep1_assets.py --data D --cache C --judges J --out O --mod img|aud --space latent|raw \
    [--steps N] [--seed S] [--rng paired] [--eval_seed E]
```

| `--mod` / `--space` | 任务名 | 每条样本进主干的样子 |
|---|---|---|
| `img` / `latent` | `t2i` | FLUX.2 潜变量 8×8 → **64 个 32 维** |
| `img` / `raw` | `t2p` | 像素 64×64×3 切成 8×8 的块 → **64 个 192 维** |
| `aud` / `latent` | `t2a` | Mimi 潜变量 26 帧,每帧拆 4 段 → **104 个 128 维** |
| `aud` / `raw` | `t2w` | 波形 49920 个采样点切成 104 段 → **104 个 480 维** |

图的那一对是第 3 集用的,声音的那一对是第 4 集的 4-1。每次只训一个版本;两版各占一张卡同时跑,耗时和显存才有可比性。

## 2. 往 sparrow 里临时登记两个任务

[ep1_assets.py:30-37](../code/ep1_assets.py#L30)

```python
S.MODS["pix"] = (192, 64, 1, True)                       # 像素版:8x8 一块,64 块,每块 8*8*3=192 个数
S.MODS["wav"] = (480, 104, 1, True)                      # 波形版:49920 个数切 104 块,每块 480 个(和 Mimi 版同样 104 个向量)
S.TEMPLATES["t2p"] = ["<bos>", "C", "A", "<img>", ("pix", True)]
S.TEMPLATES["t2w"] = ["<bos>", "A", "<aud>", ("wav", True)]
S.GEN |= {"t2p", "t2w"}
TASK = {("img", "latent"): "t2i", ("img", "raw"): "t2p", ("aud", "latent"): "t2a", ("aud", "raw"): "t2w"}
PAIR = {"t2p": "t2i", "t2w": "t2a"}
```

- 直接修改 `sparrow` 模块里的全局 `MODS` 和 `TEMPLATES`,`Layout`、`Sparrow`、`train`、`generate` 都不用改。这也是第 2 集「模板驱动」设计的好处。
- 登记发生在构造模型**之前**,所以无论跑哪一版,`Sparrow()` 都会把 `inp.pix`、`inp.wav`、`flow_head.pix`、`flow_head.wav` 一起建出来。两版的参数集合完全相同,只是各自只训练用到的那几个。
- 波形版特意切成 **104 段**,和 Mimi 版的向量数一样,所以两版序列长度都是 107。

### 2.1 像素切块:`pix_tokens` / `pix_image`

[ep1_assets.py:40-47](../code/ep1_assets.py#L40)

```python
def pix_tokens(u8):          # [B, 64, 64, 3] uint8 -> [B, 64, 192] in [-1, 1]
    return to_float(u8).view(B, 3, 8, 8, 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(B, 64, 192)
```

`to_float` 先变成 `[B, 3, 64, 64]` 并缩放到 [-1, 1],再切成 8×8 个 8×8 的块,每块把 3 个通道的 64 个像素拉平成 192 维。`pix_image` 做反操作。像素版**不做**均值/标准差归一化,只缩放到 [-1, 1]。

### 2.2 波形切段

```python
x = torch.from_numpy(train_wavs(voice)[:len(self.animal)]).float().view(len(self.animal), 104, 480)
```

直接复用 `encode.py` 的 [`train_wavs`](../code/encode.py#L30)(第 4 集为此把它提成了模块级函数),所以波形版和 Mimi 版拿到的是**同一批**时间平移后的录音:训练 324 条 × 4 = 1296 行。之后逐维减均值、除以标准差;解码时反算,再 `clamp(-1, 1)`。

## 3. 数据整份放在卡上:`OneData`

[ep1_assets.py:50-91](../code/ep1_assets.py#L50)。这个脚本不用 `sparrow.Cache`,而是自己装数据:

- 只装这一项本事需要的标签和向量,**整份放在卡上**(`summary.json` 里的 `data_on_gpu_MB` 单独记这部分)。
- `get(task, B, gen)` 的签名和 `Cache.get` 相同,所以能直接交给 `S.train`。
- `decode(x, codec)`:像素版用 `pix_image`,波形版反归一化后 clamp,潜变量版交给 FLUX.2 VAE 或 Mimi 解码。

## 4. 配对:两版只差表示

[ep1_assets.py:114-121](../code/ep1_assets.py#L114)

```python
model = S.Sparrow(seed=a.seed if a.rng == "paired" else None).to(dev)
...
args = SimpleNamespace(tasks=[data.task], steps=a.steps, lr=6e-4, bs=512, bs_vid=128, bs_task={}, ema=0.999, weights={},
                       eval_points=[], seed=a.seed, rng=a.rng, stream_alias=PAIR)
sec, log, train_peak, _ = S.train(args, model, data, lays)
```

`--rng paired` 时有两层保证(机制见 [03](03-随机协议与可复现.md)):

1. **初始化相同**:公共主干在同一种子下初始值一样,用 `init.json` 里的 `trunk_init_sha256` 核对。
2. **训练抽样相同**:`stream_alias=PAIR` 让 `t2w` 的随机流按 `t2a` 的名字取种子,两版每一步抽到的样本序号、时间 `t`、去条件都相同。

只有一样避不开:两版的加噪噪声维数不同(128 维 vs 480 维),抽出来的数自然不同。

另外,这里 `weights={}`,也就是单任务、损失权重 1;主线 sparrow.py 里 t2a 的权重是 4。

## 5. 产物

| 文件 | 内容 |
|---|---|
| `init.json` | 起跑就写,同 sparrow.py 的 `identity()` |
| `loss.csv` | 每 100 步的训练损失 |
| `summary.json` | 训练耗时、每步秒数、训练显存峰值、数据占显存、进出主干两个投影的参数量(`proj_in_params` / `proj_out_params`)、序列长度、正确率、随机协议、`stream_identity` |
| `samples/` | 声音:6 条 `gen_<space>_<k>_<动物>.wav`;图:6 张红色动物和一张横条 |

注意:这个脚本**不保存** `final.pt`,所以没法拿 infer.py 去加载它的结果。

评测题量:图 240 张(6 种动物 × 8 种颜色 × 5 张),交给图判卷器判;声音 240 条(6 × 40),交给 Paraformer 听写。起点噪声同样用 `S.eval_noise`。

## 6. 4-1 的结果

出自 [EXPERIMENTS.md](../EXPERIMENTS.md) 4-1:

```bash
python ep1_assets.py ... --out "$W/runs/audio_latent"   --mod aud --space latent --steps 8000 --seed 0 --rng paired
python ep1_assets.py ... --out "$W/runs/audio_waveform" --mod aud --space raw    --steps 8000 --seed 0 --rng paired
```

| 表示 | 念名字正确率 | 每个向量的维数 | 完整序列长度 | 训练显存峰值 |
|---|---:|---:|---:|---:|
| Mimi 连续潜变量 | 97.9%(235/240) | 128 | 107 | 7.03 GB |
| 直接波形 | 0.0%(0/240) | 480 | 107 | 7.66 GB |

潜变量版逐字全对 96.2%(231/240)。

> **适用范围**:两版的向量数相同,但每个向量的维数不同,进出主干的投影层大小也不同。所以这组结果比较的是「两种表示方案」整体,不能只归因于压缩倍率。每档也只跑了一个训练种子。

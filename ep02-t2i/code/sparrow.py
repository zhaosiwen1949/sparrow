"""小麻雀:一个 ~16M 参数的小 Transformer,同时会画图、念名字、做动画、看图/听声/看视频说出是什么、听到名字画出来。

架构照 Transfusion / BAGEL 的路子,规模缩到单卡三小时:
  - 文字和各模态的向量排成一条序列,进同一个主干;文字只看前面(因果),同一段模态内部互相都看得见。
  - 文字出口:预测下一个词。生成出口:流匹配(z_t=(1-t)·噪声+t·数据,学速度 v=数据-噪声,和第 1 讲同一套)。
  - 生成走 VAE / 音频编解码器的潜空间,理解走另一套编码器(SigLIP2、Whisper)+ 小投影层——
    「理解」和「生成」用不同的编码器,和 Janus、BAGEL、Qwen-Omni 一样。冻结零件见 frozen.py,训练前已缓存。

七种序列(* 是要预测的答案词,[] 是一段模态向量):
  文字→图   <bos> 颜色 动物 <img> [FLUX.2 潜变量 8x8=64,加噪]
  文字→声音 <bos> 动物 <aud> [Mimi 潜变量 26 帧 x 4,加噪]
  文字→视频 <bos> 颜色 动物 动作 <vid> [Wan 潜变量 5 帧 x 64,加噪]
  看图      <bos> <img> [SigLIP 16] </img> 颜色* 动物* <eos>*
  听声      <bos> <aud> [Whisper 26] </aud> 动物* <eos>*
  看视频    <bos> <vid> [SigLIP 16 帧 x 4] </vid> 颜色* 动物* 动作* <eos>*
  听→画    <bos> <aud> [Whisper 26] </aud> <img> [FLUX.2 潜变量,加噪]

  python sparrow.py --data D --cache C --judges J --out O [--tasks t2i,t2a,...] [--steps N] [--init 上一集/final.pt]
训练完自动评测、导出样例,写 O/report.json;只留最终权重 O/final.pt(含归一化统计量,infer.py 直接能用)。
--init 接着上一集的权重往下训(课程逐集加本事);--eval_tasks 评测哪些本事(默认同 --tasks,可以多评几样量「忘没忘」);
--eval_every N 训练中途每 N 步也评测一次(写 O/step<N>/);--vid_attn frame 让视频每帧只看自己(对照)。
--eval_only 读 O/final.pt 重新评测,只替换 report.json 里的评测部分(训练参数、耗时、日志照旧)。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from parts import (ACTION_ID, ACTION_ZH, ACTIONS, ANIMAL_ID, ANIMAL_ZH, ANIMALS, AUD_FRAMES, COLOR_ID, COLOR_ZH, COLORS, ID, SR, VOCAB,
                   ImageJudge, VideoJudge, load_judge, to_float)

dev = "cuda"
# 模态:(每个向量几维, 每帧几个向量, 几帧, 是生成用的潜变量吗)
MODS = {"img": (32, 64, 1, True), "vid": (16, 64, 5, True), "aud": (512, AUD_FRAMES, 1, True),
        "see": (768, 16, 1, False), "watch": (768, 4, 16, False), "hear": (384, AUD_FRAMES, 1, False)}
TASKS = ["t2i", "t2a", "t2v", "i2t", "a2t", "v2t", "a2i"]
GEN = {"t2i", "t2a", "t2v", "a2i"}
# 模板:字符串是固定词,"C"/"A"/"T" 是颜色/动物/动作的空位,末尾带 * 的是答案;元组是一段模态向量(模态, 加噪吗)
TEMPLATES = {
    "t2i": ["<bos>", "C", "A", "<img>", ("img", True)],
    "t2a": ["<bos>", "A", "<aud>", ("aud", True)],
    "t2v": ["<bos>", "C", "A", "T", "<vid>", ("vid", True)],
    "i2t": ["<bos>", "<img>", ("see", False), "</img>", "C*", "A*", "<eos>*"],
    "a2t": ["<bos>", "<aud>", ("hear", False), "</aud>", "A*", "<eos>*"],
    "v2t": ["<bos>", "<vid>", ("watch", False), "</vid>", "C*", "A*", "T*", "<eos>*"],
    "a2i": ["<bos>", "<aud>", ("hear", False), "</aud>", "<img>", ("img", True)],
}


def split_audio(k):
    """每个 Mimi 帧(512 维)拆成 k 个向量进主干。"""
    MODS["aud"] = (512 // k, AUD_FRAMES * k, 1, True)


def needs(tasks):
    """这些任务用得到哪几种向量(缓存按它分文件读)。"""
    return {x[0] for t in tasks for x in TEMPLATES[t] if isinstance(x, tuple)}


class Layout:
    """一种序列的静态结构:哪几段是文字、哪几段是模态、谁能看见谁。
    frame=True:视频段里每帧只看自己这一帧(和前面的文字),帧与帧之间互相看不见。"""

    def __init__(self, task, frame=False):
        self.task, self.pieces, tok, seg, block, answer, slots = task, [], [], [], [], [], {}
        for item in TEMPLATES[task]:
            s = len(tok)
            if isinstance(item, tuple):
                mod, noisy = item
                per, f = MODS[mod][1], MODS[mod][2]
                n = per * f
                self.pieces.append(("mod", mod, s, s + n, noisy))
                tok += [0] * n
                seg += [len(self.pieces)] * n
                block += [len(self.pieces) * 100 + (j // per if frame else 0) for j in range(n)]
                answer += [False] * n
                if noisy:
                    self.noisy = (mod, s, s + n)
            else:
                w = item.rstrip("*")
                if w in ("C", "A", "T"):
                    slots[w] = s
                tok.append(ID.get(w, 0))
                seg.append(-1)
                block.append(-1)
                answer.append(item.endswith("*"))
                if self.pieces and self.pieces[-1][0] == "text":
                    self.pieces[-1] = ("text", self.pieces[-1][1], s + 1)
                else:
                    self.pieces.append(("text", s, s + 1))
        self.L, self.slots = len(tok), slots
        self.tok = torch.tensor(tok, device=dev)
        self.answer = torch.tensor([i for i, x in enumerate(answer) if x], device=dev)
        b, p = torch.tensor(block, device=dev), torch.tensor(seg, device=dev)
        i = torch.arange(self.L, device=dev)
        same = lambda x: (x[:, None] == x[None, :]) & (x[:, None] >= 0)  # noqa: E731
        # 文字只看前面;同一块(整段模态,或 frame 时的一帧)互相都看得见;同一段模态里不同块互相看不见
        self.mask = ((i[None, :] <= i[:, None]) & ~same(p)) | same(b)


# ---------------------------------------------------------------- 主干


def sinus(t, dim=256):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t[..., None].float() * 1000 * f
    return torch.cat([a.cos(), a.sin()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.n2(x))


class Sparrow(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, maxlen=400):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), dim)
        self.pos = nn.Embedding(maxlen, dim)                                   # 序列里第几个
        self.temb = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        # 进主干:生成潜变量用一层线性;理解编码器的输出用两层小投影(LLaVA 式)
        self.inp = nn.ModuleDict({m: nn.Linear(d, dim) if g else
                                  nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dim), nn.GELU(), nn.Linear(dim, dim))
                                  for m, (d, n, f, g) in MODS.items()})
        self.where = nn.ParameterDict({m: nn.Parameter(torch.randn(n, dim) * 0.02) for m, (d, n, f, g) in MODS.items()})
        self.when = nn.ParameterDict({m: nn.Parameter(torch.randn(f, dim) * 0.02) for m, (d, n, f, g) in MODS.items() if f > 1})
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.text_head = nn.Linear(dim, len(VOCAB))                            # 文字出口
        self.flow_head = nn.ModuleDict({m: nn.Linear(dim, d) for m, (d, n, f, g) in MODS.items() if g})   # 生成出口
        for h in self.flow_head.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def embed(self, mod, x, t):
        d, n, f, g = MODS[mod]
        h = self.inp[mod](x)
        pos = self.where[mod].repeat(f, 1)                                     # 帧内位置
        if f > 1:
            pos = pos + self.when[mod].repeat_interleave(n, 0)                 # 时间位置:第几帧
        h = h + pos
        return h + self.temb(sinus(t))[:, None] if g else h

    def forward(self, lay, tok, segs, t):
        """segs:按模板顺序每段模态一个 [B, n, d];t:加噪那一段的时间 [B](理解任务传 None)。"""
        hs, k = [], 0
        for p in lay.pieces:
            if p[0] == "text":
                hs.append(self.tok(tok[:, p[1]:p[2]]))
            else:
                hs.append(self.embed(p[1], segs[k], t if p[4] else torch.ones_like(tok[:, 0], dtype=torch.float)))
                k += 1
        h = torch.cat(hs, 1) + self.pos.weight[:lay.L]
        for b in self.blocks:
            h = b(h, lay.mask)
        h = self.norm(h)
        v = None
        if hasattr(lay, "noisy"):
            m, s, e = lay.noisy
            v = self.flow_head[m](h[:, s:e])
        return self.text_head(h), v


# ---------------------------------------------------------------- 缓存的数据


def tokens(name, x):
    """缓存里的原始形状 -> [N, 向量个数, 维度]。"""
    if name == "img_lat":                   # [N, 32, 8, 8] -> [N, 64, 32]
        return x.flatten(2).transpose(1, 2)
    if name == "vid_lat":                   # [N, 16, 5, 8, 8] -> [N, 5*64, 16],按帧排
        return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1])
    if name == "aud_lat":                   # [N, 26, 512] -> [N, 26*k, 512/k]:每个 Mimi 帧拆成 k 个向量
        return x.reshape(x.shape[0], -1, MODS["aud"][0])
    return x


def untokens(name, x):
    if name == "img_lat":
        return x.transpose(1, 2).reshape(x.shape[0], 32, 8, 8)
    if name == "vid_lat":
        return x.view(x.shape[0], 5, 8, 8, 16).permute(0, 4, 1, 2, 3)
    if name == "aud_lat":
        return x.reshape(x.shape[0], AUD_FRAMES, 512)
    return x


FIELD = {"img": "img_lat", "vid": "vid_lat", "aud": "aud_lat", "see": "img_see", "watch": "vid_see", "hear": "aud_hear"}


def norm(stat, mod, x):
    """缓存里的原始形状 -> 归一化的向量 [N, 向量个数, 维度]。"""
    return (tokens(FIELD[mod], x.float()) - stat[mod][0]) / stat[mod][1]


def denorm(stat, mod, x):
    """归一化的向量 -> 冻结解码器要的原始潜变量。"""
    return untokens(FIELD[mod], x.float() * stat[mod][1] + stat[mod][0])


class Cache:
    """只读 mods 用得到的那几份缓存(encode.py 按零件分文件存)。标签放卡上;向量留在内存里,每步只把这一批搬上卡,
    所以显存只跟批大小和任务有关,跟缓存多大无关。每种向量按训练集的均值/标准差归一化(逐维,在内存里分块算)。"""

    def __init__(self, path, mods):
        self.tr, self.te, self.voice = {}, {}, {}
        split = {"train": self.tr, "test": self.te, "voice": self.voice}
        for f in ["labels", *sorted(mods)]:
            for k, v in np.load(f"{path}/{f}.npz").items():
                s, name = k.split(".", 1)
                split[s][name] = torch.from_numpy(v).to(dev if f == "labels" else "cpu")
        vtr = ~self.voice["test"]
        self.stat = {}
        for mod in sorted(mods):
            field = FIELD[mod]
            src = self.voice[field][vtr.cpu()] if field.startswith("aud") else self.tr[field]
            n, s1, s2 = 0, 0, 0
            for c in src.split(512):
                x = tokens(field, c.double()).flatten(0, 1)
                n, s1, s2 = n + len(x), s1 + x.sum(0), s2 + (x * x).sum(0)
            mean = s1 / n
            st = {mod: [mean.float(), ((s2 - n * mean * mean) / (n - 1)).sqrt().float() + 1e-4]}
            for d in (self.tr, self.te, self.voice):
                if field in d:
                    d[field] = torch.cat([norm(st, mod, c).half() for c in d[field].split(512)])
            self.stat[mod] = [x.to(dev) for x in st[mod]]
        self.vtr = torch.nonzero(vtr).squeeze(1)
        by = [torch.nonzero(self.tr["img_animal"] == k).squeeze(1) for k in range(len(ANIMALS))]
        self.count = torch.tensor([len(b) for b in by], device=dev)
        self.by_animal = torch.stack([F.pad(b, (0, int(self.count.max()) - len(b))) for b in by])

    def raw(self, mod, x):
        return denorm(self.stat, mod, x)

    def get(self, task, B):
        """取一批训练数据:返回 (词槽, 按模板顺序的各段向量,已搬上卡)。"""
        r = lambda n: torch.randint(0, n, (B,), device=dev)  # noqa: E731
        tr, v = self.tr, self.voice
        pick = lambda x, i: x[i.cpu()].to(dev)  # noqa: E731
        if task in ("t2i", "i2t"):
            i = r(len(tr["img_animal"]))
            return ({"C": COLOR_ID.to(dev)[tr["img_color"][i]], "A": ANIMAL_ID.to(dev)[tr["img_animal"][i]]},
                    [pick(tr[FIELD["img" if task == "t2i" else "see"]], i)])
        if task in ("t2v", "v2t"):
            i = r(len(tr["vid_animal"]))
            return ({"C": COLOR_ID.to(dev)[tr["vid_color"][i]], "A": ANIMAL_ID.to(dev)[tr["vid_animal"][i]],
                     "T": ACTION_ID.to(dev)[tr["vid_action"][i]]},
                    [pick(tr[FIELD["vid" if task == "t2v" else "watch"]], i)])
        i = self.vtr[r(len(self.vtr))]
        a = v["animal"][i]
        if task == "a2i":                      # 同一种动物,随便挑一张图配对
            j = self.by_animal[a, (torch.rand(B, device=dev) * self.count[a]).long()]
            return {}, [pick(v["aud_hear"], i), pick(tr["img_lat"], j)]
        return {"A": ANIMAL_ID.to(dev)[a]}, [pick(v[FIELD["aud" if task == "t2a" else "hear"]], i)]


def fill_tok(lay, slots, B, drop=None):
    tok = lay.tok[None].repeat(B, 1)
    for s, pos in lay.slots.items():
        tok[:, pos] = slots[s]
        if drop is not None:
            tok[drop, pos] = ID["<null>"]
    return tok


# ---------------------------------------------------------------- 训练


def train(a, model, data, lays):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    ema = [p.detach().clone() for p in model.parameters()]
    log, t0 = [], time.time()
    for step in range(a.steps):
        lr = a.lr * min(1, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        losses = {}
        for task in a.tasks:
            lay, B = lays[task], a.bs_vid if task in ("t2v", "v2t") else a.bs
            slots, segs = data.get(task, B)
            segs = [s.float() for s in segs]
            if task in GEN:
                t = torch.sigmoid(torch.randn(B, device=dev))                  # logit-normal:中间的 t 多练
                drop = (torch.rand(B, device=dev) < 0.1) if lay.slots else None  # 10% 去掉条件,供无条件引导用
                tok = fill_tok(lay, slots, B, drop)
                x1 = segs[-1]
                eps = torch.randn_like(x1)
                segs[-1] = (1 - t[:, None, None]) * eps + t[:, None, None] * x1
                with torch.autocast("cuda", torch.bfloat16):
                    _, v = model(lay, tok, segs, t)
                loss = F.mse_loss(v.float(), x1 - eps)
            else:
                tok = fill_tok(lay, slots, B)
                with torch.autocast("cuda", torch.bfloat16):
                    logits, _ = model(lay, tok, segs, None)
                loss = F.cross_entropy(logits[:, lay.answer - 1].float().flatten(0, 1), tok[:, lay.answer].flatten())
            (loss * a.weights.get(task, 1.0)).backward()
            if step % 100 == 0:
                losses[task] = round(loss.item(), 4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(ema, [p.detach() for p in model.parameters()], 1 - d)
        if step % 100 == 0:
            log.append({"step": step, "sec": round(time.time() - t0, 1), **losses})
            print(json.dumps(log[-1], ensure_ascii=False), flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            t0 += midway(a, model, ema, data, lays, step + 1)                # 中途评测的时间不算训练时间
    torch.cuda.synchronize()
    with torch.no_grad():
        for e, p in zip(ema, model.parameters()):
            p.copy_(e)
    return round(time.time() - t0, 1), log


def midway(a, model, ema, data, lays, step):
    """训练中途用当时的 EMA 权重评测一次,写 O/step<步数>/report.json(画「对的比例随步数」曲线);不打乱训练的随机数。
    返回评测用了几秒。"""
    rng = torch.cuda.get_rng_state(), torch.get_rng_state()
    live = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for e, p in zip(ema, model.parameters()):
            p.copy_(e)
    out = f"{a.out}/step{step}"
    t0 = time.time()
    rep = {"step": step, "eval": evaluate(a, model, data, lays, out), "eval_seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(f"{out}/report.json", "w"), ensure_ascii=False, indent=1)
    with torch.no_grad():
        for v, p in zip(live, model.parameters()):
            p.copy_(v)
    torch.cuda.set_rng_state(rng[0])
    torch.set_rng_state(rng[1])
    return rep["eval_seconds"]


def load(path):
    """final.pt -> (模型, 各任务的序列结构, 归一化统计量);不要缓存,推理(infer.py)用。"""
    ck = torch.load(path, map_location=dev)
    a = ck["args"]
    split_audio(a["aud_split"])
    model = Sparrow(a["dim"], a["depth"], a["heads"]).to(dev)
    model.load_state_dict(ck["model"])
    lays = {t: Layout(t, a["vid_attn"] == "frame") for t in TASKS}
    return model.eval(), lays, {m: [x.to(dev) for x in st] for m, st in ck["stat"].items()}


# ---------------------------------------------------------------- 生成与理解


def start_noise(lay, B, seed):
    """生成的起点:按种子固定的一团纯噪声(同种子同任务,每次都是同一团)。"""
    d, n, f, _ = MODS[lay.noisy[0]]
    return torch.randn(B, n * f, d, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))


@torch.no_grad()
def generate(model, lays, task, slots, B, cond=(), cfg=3.0, steps=30, x0=None, trace=None):
    """从纯噪声(t=0)沿速度场走到 t=1。无条件那一路:同模板把条件词换成 <null>;听→画 借用 文字→图 的无条件。
    x0:指定起点噪声;trace:给一个列表就把每一步的 x 都存进去(含起点)。"""
    lay = lays[task]
    ulay = lays["t2i"] if task == "a2i" else lay
    d, n, f, _ = MODS[lay.noisy[0]]
    x = torch.randn(B, n * f, d, device=dev) if x0 is None else x0.clone()
    if trace is not None:
        trace.append(x)
    tok = fill_tok(lay, slots, B)
    utok = fill_tok(ulay, {s: torch.full((B,), ID["<null>"], device=dev) for s in ulay.slots}, B)
    for k in range(steps):
        t = torch.full((B,), k / steps, device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            v = model(lay, tok, list(cond) + [x], t)[1].float()
            u = model(ulay, utok, [x], t)[1].float()
        x = x + (u + cfg * (v - u)) / steps
        if trace is not None:
            trace.append(x)
    return x


@torch.no_grad()
def describe(model, lays, task, segs, B):
    """逐词贪心地说出答案:因果注意力保证每个位置只看得见已经说出的词。"""
    lay = lays[task]
    tok = fill_tok(lay, {s: torch.zeros(B, dtype=torch.long, device=dev) for s in lay.slots}, B)
    for p in lay.answer.tolist():
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(lay, tok, [s.float() for s in segs], None)
        tok[:, p] = logits[:, p - 1].argmax(-1)
    return tok[:, lay.answer]


def chunks(n, size):
    return [(k, min(n, k + size)) for k in range(0, n, size)]


def evaluate(a, model, data, lays, out):
    """每样本事的对错率 + 导出样例。图和视频交给独立训的小分类器判,念名字交给语音识别 Paraformer 听写(asr.py)。"""
    import frozen
    torch.manual_seed(1234)
    ij = load_judge(ImageJudge, f"{a.judges}/judge_img.pt", dev)
    vj = load_judge(VideoJudge, f"{a.judges}/judge_vid.pt", dev)
    tasks = set(a.eval_tasks)
    raw_te = np.load(f"{a.data}/world_test.npz")
    raw_voice = np.load(f"{a.data}/voice.npz")["wav"] if tasks & {"a2t", "a2i"} else None
    te, v = data.te, data.voice
    vte = torch.nonzero(v["test"]).squeeze(1)
    yte = v["animal"][vte]
    ex = f"{out}/samples"
    os.makedirs(ex, exist_ok=True)
    rep = {}
    ids = lambda table, idx: table.to(dev)[idx]  # noqa: E731
    acc = lambda ok: ok.float().mean().item()  # noqa: E731
    big = lambda u8, s=4: Image.fromarray(np.asarray(u8.cpu() if torch.is_tensor(u8) else u8)).resize((64 * s, 64 * s), Image.NEAREST)  # noqa: E731
    word = lambda i: VOCAB[int(i)].replace("/", "")  # noqa: E731  没训过的模型会说出 </img> 之类,斜杠不能进文件名
    img_vae = frozen.ImageVAE() if tasks & {"t2i", "a2i"} else None
    vid_vae = frozen.VideoVAE() if "t2v" in tasks else None
    voice = frozen.Voice() if "t2a" in tasks else None

    def gif(frames, path):
        ims = [big(fr, 3) for fr in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0)

    def draw(task, slots, n, cond=None, size=128):
        outs = []
        for s, e in chunks(n, size):
            x = generate(model, lays, task, {k: val[s:e] for k, val in slots.items()}, e - s,
                         cond=[c[s:e].float() for c in cond] if cond else (), cfg=a.cfg)
            outs.append(x)
        return torch.cat(outs)

    if "t2i" in tasks:                            # 1 文字→图:6 动物 x 8 颜色 x 8 张
        ai = torch.arange(6, device=dev).repeat_interleave(64)
        ci = torch.arange(8, device=dev).repeat_interleave(8).repeat(6)
        x = draw("t2i", {"C": ids(COLOR_ID, ci), "A": ids(ANIMAL_ID, ai)}, len(ai))
        imgs = torch.cat([img_vae.decode(data.raw("img", x[s:e])) for s, e in chunks(len(x), 128)])
        pa, pc = ij(to_float(imgs))
        pa, pc = pa.argmax(1), pc.argmax(1)
        rep["t2i"] = {"n": len(ai), "animal_acc": acc(pa == ai), "color_acc": acc(pc == ci), "both_acc": acc((pa == ai) & (pc == ci))}
        grid = imgs.view(6, 8, 8, 64, 64, 3)[:, :, 0].permute(0, 2, 1, 3, 4).reshape(6 * 64, 8 * 64, 3)
        Image.fromarray(grid.cpu().numpy()).resize((8 * 128, 6 * 128), Image.NEAREST).save(f"{ex}/gen_t2i_grid.png")
        for k, (an, cn) in enumerate([(0, 0), (1, 4), (2, 6), (3, 2), (4, 5)]):
            big(imgs[an * 64 + cn * 8 + k]).save(f"{ex}/gen_t2i_{k}_{COLOR_ZH[COLORS[cn]]}{ANIMAL_ZH[ANIMALS[an]]}.png")
        # 没背过训练图:生成图到训练集最近邻的像素距离,对照「测试图到训练集」的最近邻距离
        tr = torch.from_numpy(np.load(f"{a.data}/world_train.npz")["img"]).flatten(1)          # 留在内存,分块搬上卡

        def nn_dist(q):
            q = q.flatten(1).float()
            return torch.stack([torch.cdist(q, tr[s:e].to(dev).float()).min(1).values for s, e in chunks(len(tr), 8192)]).min(0).values
        rep["t2i"]["nn_dist_gen_mean"] = nn_dist(imgs[:128]).mean().item()
        rep["t2i"]["nn_dist_test_mean"] = nn_dist(torch.from_numpy(raw_te["img"][:128]).to(dev)).mean().item()
        del tr

    if "t2a" in tasks:                            # 2 文字→念名字:6 x 40 条
        ai = torch.arange(6, device=dev).repeat_interleave(40)
        x = draw("t2a", {"A": ids(ANIMAL_ID, ai)}, len(ai))
        wav = torch.cat([voice.decode(data.raw("aud", x[s:e])) for s, e in chunks(len(x), 64)])
        from asr import Paraformer
        judge = Paraformer()
        same_sound, exact, heard = judge.hits(wav, SR, [ANIMAL_ZH[ANIMALS[int(k)]] for k in ai])
        rep["t2a"] = {"n": len(ai), "judge": judge.name,
                      "sound_acc": float(same_sound.mean()),                          # 读音对上(拼音不计声调),主指标
                      "exact_acc": float(exact.mean()),                               # 逐字全对
                      "sound_by_word": {ANIMAL_ZH[ANIMALS[k]]: float(same_sound[k * 40:(k + 1) * 40].mean()) for k in range(6)},
                      "misheard": [h for h, k in zip(heard, same_sound) if not k][:20]}
        for k in range(6):
            sf.write(f"{ex}/gen_t2a_{k}_{ANIMAL_ZH[ANIMALS[k]]}.wav", wav[k * 40].cpu().numpy(), SR)

    if "t2v" in tasks:                            # 3 文字→动画:6 动物 x 4 动作 x 8 段,颜色轮换
        ai = torch.arange(6, device=dev).repeat_interleave(32)
        ti = torch.arange(4, device=dev).repeat_interleave(8).repeat(6)
        ci = torch.arange(len(ai), device=dev) % 8
        x = draw("t2v", {"C": ids(COLOR_ID, ci), "A": ids(ANIMAL_ID, ai), "T": ids(ACTION_ID, ti)}, len(ai), size=64)
        vids = torch.cat([vid_vae.decode(data.raw("vid", x[s:e])) for s, e in chunks(len(x), 32)])
        pa, pc, pt = [torch.cat(o).argmax(1) for o in zip(*[vj(to_float(vids[s:e])) for s, e in chunks(len(vids), 64)])]
        rep["t2v"] = {"n": len(ai), "animal_acc": acc(pa == ai), "color_acc": acc(pc == ci), "action_acc": acc(pt == ti),
                      "all_acc": acc((pa == ai) & (pc == ci) & (pt == ti))}
        for k, j in enumerate([0, 41, 82, 123, 164]):          # 颜色、动物、动作都错开
            gif(vids[j], f"{ex}/gen_t2v_{k}_{COLOR_ZH[COLORS[int(ci[j])]]}{ANIMAL_ZH[ANIMALS[int(ai[j])]]}{ACTION_ZH[ACTIONS[int(ti[j])]]}.gif")

    if "i2t" in tasks:                            # 4a 看图说话:测试集 2000 张(种子与训练不重叠)
        n = len(te["img_animal"])
        ans = torch.cat([describe(model, lays, "i2t", [te["img_see"][s:e].to(dev)], e - s) for s, e in chunks(n, 500)])
        c_ok, a_ok = ans[:, 0] == ids(COLOR_ID, te["img_color"]), ans[:, 1] == ids(ANIMAL_ID, te["img_animal"])
        rep["i2t"] = {"n": n, "color_acc": acc(c_ok), "animal_acc": acc(a_ok), "both_acc": acc(c_ok & a_ok)}
        for k in range(5):
            big(raw_te["img"][k]).save(f"{ex}/i2t_{k}_模型说{word(ans[k, 0])}{word(ans[k, 1])}.png")

    if "a2t" in tasks:                            # 4b 听声说名字:没听过的 4 个音色
        ans = describe(model, lays, "a2t", [v["aud_hear"][vte.cpu()].to(dev)], len(vte))
        rep["a2t"] = {"n": len(vte), "animal_acc": acc(ans[:, 0] == ids(ANIMAL_ID, yte))}
        for k in range(5):
            j = k * len(vte) // 5
            sf.write(f"{ex}/a2t_{k}_模型说{word(ans[j, 0])}.wav", raw_voice[int(v["wav_index"][vte[j]])], SR)

    if "v2t" in tasks:                            # 4c 看视频说出颜色、动物和动作
        n = len(te["vid_animal"])
        ans = torch.cat([describe(model, lays, "v2t", [te["vid_see"][s:e].to(dev)], e - s) for s, e in chunks(n, 250)])
        c_ok = ans[:, 0] == ids(COLOR_ID, te["vid_color"])
        a_ok = ans[:, 1] == ids(ANIMAL_ID, te["vid_animal"])
        t_ok = ans[:, 2] == ids(ACTION_ID, te["vid_action"])
        rep["v2t"] = {"n": n, "color_acc": acc(c_ok), "animal_acc": acc(a_ok), "action_acc": acc(t_ok), "all_acc": acc(c_ok & a_ok & t_ok)}
        for k in range(3):
            gif(raw_te["vid"][k], f"{ex}/v2t_{k}_模型说{word(ans[k, 0])}{word(ans[k, 1])}{word(ans[k, 2])}.gif")

    if "a2i" in tasks:                            # 5 听到名字画出来:没听过的音色
        x = draw("a2i", {}, len(vte), cond=[v["aud_hear"][vte.cpu()].to(dev)])
        imgs = torch.cat([img_vae.decode(data.raw("img", x[s:e])) for s, e in chunks(len(x), 128)])
        pa = ij(to_float(imgs))[0].argmax(1)
        rep["a2i"] = {"n": len(vte), "animal_acc": acc(pa == yte)}
        for k in range(5):
            j = k * len(vte) // 5
            name = ANIMAL_ZH[ANIMALS[int(yte[j])]]
            sf.write(f"{ex}/in_a2i_{k}_听{name}.wav", raw_voice[int(v["wav_index"][vte[j]])], SR)
            big(imgs[j]).save(f"{ex}/gen_a2i_{k}_听{name}_画.png")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--eval_tasks", default="", help="评测哪些本事;不写就是 --tasks")
    ap.add_argument("--init", help="从这份 final.pt 的权重接着训(上一集);dim/depth/heads/aud_split 须一致")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--bs_vid", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--aud_split", type=int, default=4, help="每个 Mimi 帧(512 维)拆成几个向量进主干")
    ap.add_argument("--vid_attn", default="full", choices=["full", "frame"], help="视频段内:各帧互相看得见 / 每帧只看自己")
    ap.add_argument("--weights", default="t2a=4", help="各任务损失的权重;没写的都是 1")
    ap.add_argument("--eval_every", type=int, default=0, help="每训这么多步评测一次(0=只在最后评测)")
    ap.add_argument("--eval_only", action="store_true")
    a = ap.parse_args()
    a.tasks = a.tasks.split(",")
    a.eval_tasks = a.eval_tasks.split(",") if a.eval_tasks else a.tasks
    a.weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    split_audio(a.aud_split)
    t0 = time.time()
    data = Cache(a.cache, needs(a.tasks + a.eval_tasks))
    lays = {t: Layout(t, a.vid_attn == "frame") for t in TASKS}
    model = Sparrow(a.dim, a.depth, a.heads).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        bad = {k: (ck["args"][k], vars(a)[k]) for k in ("dim", "depth", "heads", "aud_split") if ck["args"][k] != vars(a)[k]}
        if bad:
            raise SystemExit(f"--init {a.init} 的结构和这次不一致(上一集, 这次):{bad}")
        model.load_state_dict(ck["model"])
    rep = {"args": vars(a), "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "seq_len": {t: lays[t].L for t in TASKS}, "load_seconds": round(time.time() - t0, 1)}
    if a.eval_only:
        model.load_state_dict(torch.load(f"{a.out}/final.pt", map_location=dev)["model"])
        rep = json.load(open(f"{a.out}/report.json"))
    else:
        rep["train_seconds"], rep["log"] = train(a, model, data, lays)
        rep["train_peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        torch.save({"model": model.state_dict(), "args": vars(a),
                    "stat": {m: [x.cpu() for x in st] for m, st in data.stat.items()}}, f"{a.out}/final.pt")
    model.eval()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    rep["eval"] = evaluate(a, model, data, lays, a.out)
    rep["eval_seconds"] = round(time.time() - t0, 1)
    rep["eval_peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    json.dump(rep, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: val for k, val in rep.items() if k != "log"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

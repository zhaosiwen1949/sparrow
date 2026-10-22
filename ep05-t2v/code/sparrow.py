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
--seed 训练种子(初始化与每步抽样);--eval_seed 评测噪声种子:每个任务第 i 题的起点噪声按 eval_seed*10000+i 单独生成,
和别的任务、别的题用了多少随机数无关,同一种子下不同权重的同一道题从同一团噪声出发。
--eval_only --ckpt 某份/final.pt 只评测那份权重:结构与训练子集照它训练时的参数,报告与样例写进 --out(不能是训练目录),
另记权重哈希、CFG 与评测种子。对照实验用的其余参数见 main() 的说明。
"""
import argparse
import hashlib
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

dev = "cuda" if torch.cuda.is_available() else "cpu"   # 没有卡时在 CPU 上跑(只用于冒烟)
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


BASE_TEMPLATES = {k: list(v) for k, v in TEMPLATES.items()}


def split_audio(k):
    """每个 Mimi 帧(512 维)拆成 k 个向量进主干。"""
    MODS["aud"] = (512 // k, AUD_FRAMES * k, 1, True)


# 结构与训练子集的参数:训练时写进 final.pt,--init / --eval_only / load 照它重建;旧权重没有的键按这里的默认值
DEFAULTS = {"dim": 384, "depth": 8, "heads": 6, "aud_split": 4, "vid_attn": "full", "maxlen": 400, "watch_frames": 16,
            "see_from": "see", "vid_lat": "vid", "per_combo": 0, "hold": "", "match_total": False, "voices": 0, "voice_draw": 0,
            "clips": 36}


def with_defaults(args):
    return {**DEFAULTS, **{k: v for k, v in args.items() if v is not None}}


def configure(a):
    """按参数定各模态的形状与模板(a 是参数字典)。可选的两种表示只在用到时登记,不改动其余权重的结构:
    see_from=img:看图改用 FLUX.2 潜变量(64 个 32 维,经同样的两层投影);vid_lat=vidimg:生视频改用逐帧 FLUX.2 潜变量(16 帧 x 64)。"""
    split_audio(a["aud_split"])
    MODS["watch"] = (768, 4, a["watch_frames"], False)
    if a["see_from"] == "img":
        MODS["seelat"] = (32, 64, 1, False)
        TEMPLATES["i2t"] = ["<bos>", "<img>", ("seelat", False), "</img>", "C*", "A*", "<eos>*"]
    if a["vid_lat"] == "vidimg":
        MODS["vidimg"] = (32, 64, 16, True)
        TEMPLATES["t2v"] = ["<bos>", "C", "A", "T", "<vid>", ("vidimg", True)]


def watch_index(n):
    """看视频取哪几帧:16 帧均分成 n 段,各取段中点那一帧(n=1 取第 8 帧,n=16 全取)。"""
    return [(2 * k + 1) * 16 // (2 * n) for k in range(n)]


def combos(spec):
    """'red-rabbit,blue-cat' -> [(颜色序号, 动物序号), ...]。"""
    return [(COLORS.index(c), ANIMALS.index(an)) for c, an in (x.split("-") for x in spec.split(",") if x)]


def pick_images(color, animal, per_combo=0, hold="", match_total=False):
    """训练用哪几张图(序号,按原顺序)。per_combo=K:每种颜色x动物只取前 K 张;hold:去掉这些组合的全部图;
    hold + match_total:那些组合照留,改为从全部训练图里随机去掉同样多张(随机数固定为 0,和训练种子无关)。"""
    keep = np.ones(len(color), bool)
    combo = color * len(ANIMALS) + animal
    if per_combo:
        rank = np.zeros(len(color), int)
        for c in np.unique(combo):
            i = np.nonzero(combo == c)[0]
            rank[i] = np.arange(len(i))
        keep &= rank < per_combo
    if hold:
        held = np.isin(combo, [c * len(ANIMALS) + an for c, an in combos(hold)]) & keep
        if match_total:
            keep[np.random.default_rng(0).choice(np.nonzero(keep)[0], held.sum(), replace=False)] = False
        else:
            keep &= ~held
    return np.nonzero(keep)[0]


def pick_voices(names, animals, test, n, draw, clips):
    """只用 n 个训练音色、共 clips 条录音(每种动物一样多),返回原录音序号。抽法 draw 决定挑哪几个音色(从全部训练音色里随机排)、
    各取哪几条;每种动物的名额在挑中的音色间轮流分配,所以每个音色分到的条数尽量相同。"""
    rng = np.random.default_rng(draw)
    chosen = list(rng.permutation(sorted(set(names[~test].tolist())))[:n])
    per, picked = clips // len(ANIMALS), []
    for an in range(len(ANIMALS)):
        for j in range(per):
            v = chosen[(an * per + j) % n]
            cand = [i for i in np.nonzero((names == v) & (animals == an) & ~test)[0] if i not in picked]
            picked.append(int(rng.choice(cand)))
    return sorted(picked), chosen


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


def stream_seed(seed, *names):
    """新随机协议里每条随机流的种子:由训练种子和流的名字决定,和别的流用了多少随机数无关。"""
    return int(hashlib.sha256(":".join(map(str, (seed, *names))).encode()).hexdigest()[:15], 16)


TRUNK = ("tok.", "pos.", "temb.", "blocks.", "norm.", "text_head.")      # 公共主干:各配置都有、同名同形的部分


def params_sha256(state, prefixes=None):
    """参数的哈希(按名字排序,名字、形状、float32 字节都算进去);prefixes 给了就只算这些前缀的参数。"""
    h = hashlib.sha256()
    for k in sorted(state):
        if prefixes is None or k.startswith(prefixes):
            v = state[k].detach().float().cpu().contiguous()
            h.update(f"{k}{tuple(v.shape)}".encode())
            h.update(v.numpy().tobytes())
    return h.hexdigest()


class Sparrow(nn.Module):
    """seed=None:旧协议,所有部件按构造顺序吃全局随机数(和九月的权重同一个初始化);
    给了 seed(新协议):每个部件在自己的随机流里初始化,同名同形的部件在任何配置下都一样,公共主干不随可选模块变化。"""

    def __init__(self, dim=384, depth=8, heads=6, maxlen=400, seed=None):
        super().__init__()

        def make(name, build):
            if seed is None:
                return build()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(stream_seed(seed, "init", name))
                return build()

        def emb(n):
            def build():
                e = nn.Embedding(n, dim)
                if seed is not None:                      # 旧协议在最后统一重设(见下),顺序照旧
                    nn.init.normal_(e.weight, std=0.02)
                return e
            return build

        def head(d):
            def build():
                h = nn.Linear(dim, d)
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
                return h
            return build
        self.tok = make("tok", emb(len(VOCAB)))
        self.pos = make("pos", emb(maxlen))                                    # 序列里第几个
        self.temb = make("temb", lambda: nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)))
        # 进主干:生成潜变量用一层线性;理解编码器的输出用两层小投影(LLaVA 式)
        self.inp = nn.ModuleDict({m: make(f"inp.{m}", lambda d=d, g=g: nn.Linear(d, dim) if g else
                                          nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dim), nn.GELU(), nn.Linear(dim, dim)))
                                  for m, (d, n, f, g) in MODS.items()})
        self.where = nn.ParameterDict({m: make(f"where.{m}", lambda n=n: nn.Parameter(torch.randn(n, dim) * 0.02))
                                       for m, (d, n, f, g) in MODS.items()})
        self.when = nn.ParameterDict({m: make(f"when.{m}", lambda f=f: nn.Parameter(torch.randn(f, dim) * 0.02))
                                      for m, (d, n, f, g) in MODS.items() if f > 1})
        self.blocks = make("blocks", lambda: nn.ModuleList(Block(dim, heads) for _ in range(depth)))
        self.norm = make("norm", lambda: nn.LayerNorm(dim))
        self.text_head = make("text_head", lambda: nn.Linear(dim, len(VOCAB)))  # 文字出口
        self.flow_head = nn.ModuleDict({m: make(f"flow_head.{m}", head(d)) for m, (d, n, f, g) in MODS.items() if g})   # 生成出口
        if seed is None:
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
    if name == "vid_img":                   # [N, 16, 32, 8, 8] -> [N, 16*64, 32],按帧排
        return x.flatten(3).transpose(2, 3).reshape(x.shape[0], -1, x.shape[2])
    return x


def untokens(name, x):
    if name == "img_lat":
        return x.transpose(1, 2).reshape(x.shape[0], 32, 8, 8)
    if name == "vid_lat":
        return x.view(x.shape[0], 5, 8, 8, 16).permute(0, 4, 1, 2, 3)
    if name == "aud_lat":
        return x.reshape(x.shape[0], AUD_FRAMES, 512)
    if name == "vid_img":
        return x.view(x.shape[0], 16, 64, 32).transpose(2, 3).reshape(x.shape[0], 16, 32, 8, 8)
    return x


FIELD = {"img": "img_lat", "vid": "vid_lat", "aud": "aud_lat", "see": "img_see", "watch": "vid_see", "hear": "aud_hear",
         "seelat": "img_lat", "vidimg": "vid_img"}
FILE = {"img": "img", "vid": "vid", "aud": "aud", "see": "see", "watch": "watch", "hear": "hear", "seelat": "img", "vidimg": "vidimg"}


def norm(stat, mod, x):
    """缓存里的原始形状 -> 归一化的向量 [N, 向量个数, 维度]。"""
    return (tokens(FIELD[mod], x.float()) - stat[mod][0]) / stat[mod][1]


def denorm(stat, mod, x):
    """归一化的向量 -> 冻结解码器要的原始潜变量。"""
    return untokens(FIELD[mod], x.float() * stat[mod][1] + stat[mod][0])


def task_mods(task, a):
    """这份配置(参数字典 a)下这个任务用哪几种向量;不动全局模板,判断上一集用过哪些模态时用。"""
    if task == "i2t" and a["see_from"] == "img":
        return ["seelat"]
    if task == "t2v" and a["vid_lat"] == "vidimg":
        return ["vidimg"]
    return [x[0] for x in BASE_TEMPLATES[task] if isinstance(x, tuple)]


def run_mods(a):
    """一份权重训练与评测时用到的全部模态(a 是它的参数字典)。"""
    return {m for t in a["tasks"] + (a.get("eval_tasks") or []) for m in task_mods(t, a)}


class Cache:
    """只读 mods 用得到的那几份缓存(encode.py 按零件分文件存)。标签放卡上;向量留在内存里,每步只把这一批搬上卡,
    所以显存只跟批大小和任务有关,跟缓存多大无关。
    先定训练子集(a 是参数字典:per_combo / hold 挑图,voices 挑录音,watch_frames 挑帧;voice 是载入的 voice.npz),
    再逐个模态定归一化统计量(逐维减均值除标准差),最后才归一化:
      stat=None                 全部按这次的训练子集现算(只用子集里的行);
      stat 给了、strict=False    接着训(--init):上一集的权重里有的照它(形状不对就报错);上一集本来就用过的模态(inherited)
                                没有统计量就报错;只有上一集没用过的(本集新加的)才按子集现算;
      strict=True               评测某份权重(--eval_only):stat 必须是字典,要用的每种都必须在它里面且形状对,缺一样就报错。
    两种模态共用一份缓存(看图的 FLUX.2 潜变量 seelat 与生图的 img)时,统计量相同才共用一份归一化后的数据,
    不同就各存一份;self.key[模态] 是这份数据在 tr / te / voice 里的键。每种统计量的来源写进 self.stat_source。"""

    def __init__(self, path, mods, a, voice=None, stat=None, strict=False, inherited=()):
        if strict and not isinstance(stat, dict):
            raise SystemExit("评测要用这份权重训练时的归一化统计量,它没有存(或存的不是字典)")
        if stat is not None and not isinstance(stat, dict):
            raise SystemExit("上一集权重里的归一化统计量不是字典")
        self.tr, self.te, self.voice = {}, {}, {}
        split = {"train": self.tr, "test": self.te, "voice": self.voice}
        for f in ["labels", *sorted({FILE[m] for m in mods})]:
            for k, v in np.load(f"{path}/{f}.npz").items():
                s, name = k.split(".", 1)
                split[s][name] = torch.from_numpy(v).to(dev if f == "labels" else "cpu")
        tr, self.subset = self.tr, {}
        if "watch" in mods:                                  # [N, 16 帧 x 4, 768] -> 只留取到的那几帧(16 帧也写明全取)
            idx = watch_index(a["watch_frames"])
            if len(idx) != 16:
                for d in (self.tr, self.te):
                    d["vid_see"] = d["vid_see"].view(len(d["vid_see"]), 16, 4, -1)[:, idx].flatten(1, 2)
            self.subset["watch_frames"] = idx
        self.img = torch.from_numpy(pick_images(tr["img_color"].cpu().numpy(), tr["img_animal"].cpu().numpy(),
                                                a["per_combo"], a["hold"], a["match_total"])).to(dev)
        if a["per_combo"] or a["hold"]:
            self.subset["images"] = {"n": len(self.img), "of": len(tr["img_animal"]), "per_combo": a["per_combo"],
                                     "hold": a["hold"], "match_total": a["match_total"]}
        test = self.voice["test"].cpu().numpy()
        rows = np.nonzero(~test)[0]
        if a["voices"]:                                      # 训练录音只留挑中的那几条(每条的 AUG 份一起留)
            names = voice["voice"]
            picked, chosen = pick_voices(names, voice["animal"], voice["test"], a["voices"], a["voice_draw"], a["clips"])
            rows = np.nonzero(~test & np.isin(self.voice["wav_index"].cpu().numpy(), picked))[0]
            self.subset["voices"] = {"voices": [str(v) for v in chosen], "voice_draw": a["voice_draw"], "clips": len(picked),
                                     "clip_names": [str(voice["name"][i]) for i in picked], "cache_rows": len(rows),
                                     "per_voice": {str(v): int(sum(names[i] == v for i in picked)) for v in chosen}}
        self.vtr = torch.from_numpy(rows).to(dev)
        # 1) 逐个模态定统计量与来源
        chosen, computed = {}, {}
        for mod in sorted(mods):
            field, want = FIELD[mod], (MODS[mod][0],)
            if stat is not None and mod in stat:
                got = [tuple(x.shape) for x in stat[mod]]
                if got != [want, want]:
                    raise SystemExit(f"{mod} 的归一化统计量形状是 {got},这次要 {want}")
                chosen[mod] = ([x.cpu().float() for x in stat[mod]], "ckpt")
            elif strict:
                raise SystemExit(f"这份权重没有存 {mod} 的归一化统计量,不能按它训练时的口径评测")
            elif mod in inherited:
                raise SystemExit(f"上一集本来就用 {mod},它的权重里却没有 {mod} 的归一化统计量;不现算顶替")
            else:
                if field not in computed:
                    if field.startswith("aud"):
                        sel, what = self.voice[field][self.vtr.cpu()], f"训练录音 {len(self.vtr)} 行"
                    elif field.startswith("img"):
                        sel, what = tr[field][self.img.cpu()], f"训练图 {len(self.img)} 张"
                    else:
                        sel, what = tr[field], f"训练动画 {len(tr[field])} 段"
                    n, s1, s2 = 0, 0, 0
                    for c in sel.split(512):
                        x = tokens(field, c.double()).flatten(0, 1)
                        n, s1, s2 = n + len(x), s1 + x.sum(0), s2 + (x * x).sum(0)
                    mean = s1 / n
                    computed[field] = ([mean.float(), ((s2 - n * mean * mean) / (n - 1)).sqrt().float() + 1e-4], f"现算:{what}")
                st, what = computed[field]
                chosen[mod] = (st, what + ("(本集新加的模态)" if stat is not None else ""))
        # 2) 同一份缓存:统计量逐元素相同的模态共用一份归一化数据,不同的各存一份
        self.stat, self.stat_source, self.key = {}, {}, {}
        for field in sorted({FIELD[m] for m in mods}):
            group = [m for m in sorted(mods) if FIELD[m] == field]
            same = all(all(torch.equal(x, y) for x, y in zip(chosen[m][0], chosen[group[0]][0])) for m in group)
            for d in (self.tr, self.te, self.voice):
                if field in d:
                    raw = d.pop(field)
                    for m in group[:1] if same else group:
                        d[field if same else f"{field}@{m}"] = torch.cat([norm({m: chosen[m][0]}, m, c).half() for c in raw.split(512)])
            for m in group:
                self.key[m] = field if same else f"{field}@{m}"
                self.stat[m], self.stat_source[m] = [x.to(dev) for x in chosen[m][0]], chosen[m][1]
        by = [self.img[tr["img_animal"][self.img] == k] for k in range(len(ANIMALS))]
        self.count = torch.tensor([len(b) for b in by], device=dev)
        self.by_animal = torch.stack([F.pad(b, (0, int(self.count.max()) - len(b))) for b in by])

    def raw(self, mod, x):
        return denorm(self.stat, mod, x)

    def get(self, task, B, gen=None):
        """取一批训练数据:返回 (词槽, 按模板顺序的各段向量,已搬上卡)。gen:这个任务抽样用的随机流(旧协议为 None,用全局随机数)。"""
        r = lambda n: torch.randint(0, n, (B,), device=dev, generator=gen)  # noqa: E731
        tr, v, key = self.tr, self.voice, self.key
        pick = lambda x, i: x[i.cpu()].to(dev)  # noqa: E731
        mods = [x[0] for x in TEMPLATES[task] if isinstance(x, tuple)]
        if task in ("t2i", "i2t"):
            i = self.img[r(len(self.img))]
            return ({"C": COLOR_ID.to(dev)[tr["img_color"][i]], "A": ANIMAL_ID.to(dev)[tr["img_animal"][i]]},
                    [pick(tr[key[mods[0]]], i)])
        if task in ("t2v", "v2t"):
            i = r(len(tr["vid_animal"]))
            return ({"C": COLOR_ID.to(dev)[tr["vid_color"][i]], "A": ANIMAL_ID.to(dev)[tr["vid_animal"][i]],
                     "T": ACTION_ID.to(dev)[tr["vid_action"][i]]},
                    [pick(tr[key[mods[0]]], i)])
        i = self.vtr[r(len(self.vtr))]
        a = v["animal"][i]
        if task == "a2i":                      # 同一种动物,随便挑一张图配对
            j = self.by_animal[a, (torch.rand(B, device=dev, generator=gen) * self.count[a]).long()]
            return {}, [pick(v[key["hear"]], i), pick(tr[key["img"]], j)]
        return {"A": ANIMAL_ID.to(dev)[a]}, [pick(v[key[mods[0]]], i)]


def fill_tok(lay, slots, B, drop=None):
    tok = lay.tok[None].repeat(B, 1)
    for s, pos in lay.slots.items():
        tok[:, pos] = slots[s]
        if drop is not None:
            tok[drop, pos] = ID["<null>"]
    return tok


# ---------------------------------------------------------------- 训练


def batch(a, task):
    """这个任务每步取几条:--bs_task 单独给的优先,否则视频两样用 --bs_vid、其余用 --bs。"""
    return a.bs_task[task] if task in a.bs_task else (a.bs_vid if task in ("t2v", "v2t") else a.bs)


class Streams:
    """新随机协议的训练随机流:每个任务的抽样、时间 t、去掉条件、训练噪声各一条独立的 Generator。
    改一个任务的批量、加减伙伴任务、换表示维度,都不改别的任务(或同任务别的用途)抽到的随机数。
    alias 给出「配对身份」:同一道题换了表示(如像素版 t2p 对潜变量版 t2i),随机流按配对身份取,两版抽到同样的样本、时间、去条件。"""

    def __init__(self, seed, tasks, alias=None):
        self.alias = alias or {}
        self.g = {(t, u): torch.Generator(device=dev).manual_seed(stream_seed(seed, "train", self.alias.get(t, t), u))
                  for t in tasks for u in ("pick", "t", "drop", "eps")}

    def __call__(self, task, use):
        return self.g[(task, use)]


def peak():
    return torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0


def reset_peak():
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()


def train(a, model, data, lays):
    """返回 (训练秒数, 每 100 步的损失记录, 训练显存峰值 GB, 中途评测显存峰值 GB);中途评测的时间与显存都不算进训练的。"""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    ema = [p.detach().clone() for p in model.parameters()]
    rs = Streams(a.seed, a.tasks, getattr(a, "stream_alias", None)) if a.rng == "paired" else (lambda task, use: None)
    log, t0, train_peak, eval_peak = [], time.time(), 0.0, 0.0
    for step in range(getattr(a, "stop_at", 0) or a.steps):
        lr = a.lr * min(1, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        losses = {}
        for task in a.tasks:
            lay, B = lays[task], batch(a, task)
            slots, segs = data.get(task, B, rs(task, "pick"))
            segs = [s.float() for s in segs]
            if task in GEN:
                t = torch.sigmoid(torch.randn(B, device=dev, generator=rs(task, "t")))                  # logit-normal:中间的 t 多练
                drop = (torch.rand(B, device=dev, generator=rs(task, "drop")) < 0.1) if lay.slots else None  # 10% 去掉条件
                tok = fill_tok(lay, slots, B, drop)
                x1 = segs[-1]
                g = rs(task, "eps")
                eps = torch.randn_like(x1) if g is None else torch.randn(x1.shape, device=dev, generator=g)
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
        if step + 1 in a.eval_points and step + 1 < (getattr(a, "stop_at", 0) or a.steps):
            train_peak = max(train_peak, peak())                              # 到这里为止的训练峰值
            reset_peak()
            t0 += midway(a, model, ema, data, lays, step + 1)                # 中途评测的时间不算训练时间
            eval_peak = max(eval_peak, peak())                                # 只含这次中途评测
            reset_peak()
    if dev == "cuda":
        torch.cuda.synchronize()
    train_peak = max(train_peak, peak())
    with torch.no_grad():
        for e, p in zip(ema, model.parameters()):
            p.copy_(e)
    return round(time.time() - t0, 1), log, round(train_peak, 2), round(eval_peak, 2)


def save(model, a, data, path):
    torch.save({"model": model.state_dict(), "args": vars(a),
                "stat": {m: [x.cpu() for x in st] for m, st in data.stat.items()}}, path)


def midway(a, model, ema, data, lays, step):
    """训练中途用当时的 EMA 权重评测一次,写 O/step<步数>/report.json(画「对的比例随步数」曲线);
    --keep_midway 时另存这时的 EMA 权重 O/step<步数>/final.pt(infer.py 能直接用)。不打乱训练的随机数。返回评测用了几秒。"""
    rng = torch.cuda.get_rng_state() if dev == "cuda" else None, torch.get_rng_state()
    live = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for e, p in zip(ema, model.parameters()):
            p.copy_(e)
    out = f"{a.out}/step{step}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rep = {"step": step, "eval": evaluate(a, model, data, lays, out), "eval_seconds": round(time.time() - t0, 1)}
    write_json(rep, f"{out}/report.json")
    if a.keep_midway:
        save(model, a, data, f"{out}/final.pt")
    with torch.no_grad():
        for v, p in zip(live, model.parameters()):
            p.copy_(v)
    if rng[0] is not None:
        torch.cuda.set_rng_state(rng[0])
    torch.set_rng_state(rng[1])
    return rep["eval_seconds"]


def load(path):
    """final.pt -> (模型, 各任务的序列结构, 归一化统计量);不要缓存,推理(infer.py)用。"""
    ck = torch.load(path, map_location=dev)
    a = with_defaults(ck["args"])
    configure(a)
    model = Sparrow(a["dim"], a["depth"], a["heads"], a["maxlen"]).to(dev)
    model.load_state_dict(ck["model"])
    lays = {t: Layout(t, a["vid_attn"] == "frame") for t in TASKS}
    return model.eval(), lays, {m: [x.to(dev) for x in st] for m, st in ck["stat"].items()}


# ---------------------------------------------------------------- 生成与理解


def start_noise(lay, B, seed):
    """生成的起点:按种子固定的一团纯噪声(同种子同任务,每次都是同一团)。"""
    d, n, f, _ = MODS[lay.noisy[0]]
    return torch.randn(B, n * f, d, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))


def eval_noise(lay, n, seed, first=0):
    """评测的起点噪声:第 i 题(从 first 起编号)单独按种子 seed*10000+i 生成,和别的题、别的任务用了多少随机数无关。"""
    return torch.cat([start_noise(lay, 1, seed * 10000 + i) for i in range(first, first + n)])


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


def topk_rmse(q, base, k):
    """q 里每张图到 base 里最像的 k 张的距离(每个像素通道的均方根误差,0~255)与序号,从近到远;base 留在内存,分块搬上卡。"""
    q = q.flatten(1).float()
    best_d = torch.full((len(q), k), float("inf"), device=dev)
    best_i = torch.zeros((len(q), k), dtype=torch.long, device=dev)
    for s, e in chunks(len(base), 8192):
        d = torch.cat([best_d, torch.cdist(q, base[s:e].to(dev).flatten(1).float())], 1)
        i = torch.cat([best_i, torch.arange(s, e, device=dev).expand(len(q), -1)], 1)
        best_d, j = d.topk(k, largest=False)
        best_i = i.gather(1, j)
    return best_d / q.shape[1] ** 0.5, best_i


def rmse_to(q, base):
    d, i = topk_rmse(q, base, 1)
    return d[:, 0], i[:, 0]


def paired_test(raw_te, animal, color):
    """给每张生成图配一张同颜色同动物的独立测试图:各组合按测试集顺序取,第 k 次用到该组合取第 k 张(不够时从头轮)。"""
    ta, tc, used, idx = raw_te["img_animal"], raw_te["img_color"], {}, []
    for an, c in zip(animal.tolist(), color.tolist()):
        pool = np.nonzero((ta == an) & (tc == c))[0]
        k = used.get((an, c), 0)
        used[(an, c)] = k + 1
        idx.append(int(pool[k % len(pool)]))
    return np.array(idx)


def nearest(gen, animal, color, train_u8, raw_te, ex, tag):
    """最近邻:生成图与按颜色x动物配对的同样张数的独立测试图,各自到「这次实际用的训练图」(train_u8)的最近距离。
    报分布与中位数之比;存最像训练图的 8 张生成图各配它最近的训练图(ex/nearest_<tag>.png,左生成、右训练)与全部距离(.csv)。"""
    ti = paired_test(raw_te, animal, color)
    dg, ig = rmse_to(gen, train_u8)
    dt, _ = rmse_to(torch.from_numpy(raw_te["img"][ti]).to(dev), train_u8)
    pct = lambda d: {str(p): round(float(np.percentile(d.cpu().numpy(), p)), 2) for p in (0, 5, 25, 50, 75, 95, 100)}  # noqa: E731
    med = lambda d: float(np.percentile(d.cpu().numpy(), 50))  # noqa: E731  中位数与分布里的 50 同一种算法
    sheet = Image.new("RGB", (2 * 196, 8 * 196), (255, 255, 255))
    for r, j in enumerate(dg.argsort()[:8].tolist()):
        for c, im in enumerate((gen[j], train_u8[int(ig[j])])):
            sheet.paste(Image.fromarray(np.asarray(im.cpu())).resize((192, 192), Image.NEAREST), (c * 196, r * 196))
    sheet.save(f"{ex}/nearest_{tag}.png")
    with open(f"{ex}/nearest_{tag}.csv", "w") as f:
        f.write("i,animal,color,rmse_gen,nearest_train_row,test_index,rmse_test\n")
        for i in range(len(gen)):
            f.write(f"{i},{ANIMALS[int(animal[i])]},{COLORS[int(color[i])]},{dg[i]:.2f},{int(ig[i])},{ti[i]},{dt[i]:.2f}\n")
    return {"metric": "每个像素通道的均方根误差(整图 L2 / sqrt(12288),0~255)", "train_searched": len(train_u8), "n": len(gen),
            "generated": pct(dg), "test": pct(dt), "median_ratio_gen_over_test": round(med(dg) / med(dt), 4)}


def video_rulers(v, ai, ci, ti, ij):
    """一批动画 [B, 16, 64, 64, 3](uint8)的两把尺子:相邻两帧的像素变化(每像素通道均方根误差,按动作取平均成 15 个点的曲线);
    逐帧交给图判卷器,16 帧都判成该动物和颜色才算这一段一致(另报逐帧对的比例与按帧号的对的比例)。"""
    acc = lambda ok: ok.float().mean().item()  # noqa: E731
    diff = torch.cat([((v[s:e, 1:].float() - v[s:e, :-1].float()) ** 2).mean((2, 3, 4)).sqrt() for s, e in chunks(len(v), 32)])
    pa, pc = [torch.cat(o).argmax(1).view(len(v), 16) for o in
              zip(*[ij(to_float(v[s:e].flatten(0, 1))) for s, e in chunks(len(v), 16)])]
    ok = (pa == ai[:, None]) & (pc == ci[:, None])
    return {"adjacent_rmse_by_action": {ACTIONS[k]: [round(x, 3) for x in diff[ti == k].mean(0).tolist()]
                                        for k in range(len(ACTIONS)) if (ti == k).any()},
            "frame_acc": acc(ok), "frame_consistent": acc(ok.all(1)), "frame_acc_by_index": [round(x, 4) for x in ok.float().mean(0).tolist()]}


VALID_FRAME_ACC = 0.97          # 图判卷器在真动画单帧上的正确率不到这个数,逐帧一致率就标无效(照出,不删)
_calib = {}


def video_calibration(raw_te, ij, vj, codec, name):
    """逐帧一致率要同时看判卷器本身可不可靠:全部测试真动画(500 段)与「压缩器编码再解码」的往返片,用同样两把尺子与视频判卷器各量一遍;
    codec 是 (encode, decode),name 标明是哪个压缩器。同一进程里同一压缩器只算一次。"""
    if name in _calib:
        return _calib[name]
    acc = lambda ok: ok.float().mean().item()  # noqa: E731
    real = torch.from_numpy(raw_te["vid"]).to(dev)
    ra, rc, rt = [torch.from_numpy(raw_te[k]).to(dev) for k in ("vid_animal", "vid_color", "vid_action")]
    enc, dec = codec
    parts = []
    for s, e in chunks(len(real), 25):            # 编码同 encode.py(bf16),解码同评测(fp32)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            z = enc(real[s:e])
        parts.append(dec(z.float()))
    rtv = torch.cat(parts)
    out = {"n": len(real), "codec": name, "threshold_real_frame_acc": VALID_FRAME_ACC}
    for key, v in (("real", real), ("roundtrip", rtv)):
        pa, pc, pt = [torch.cat(o).argmax(1) for o in zip(*[vj(to_float(v[s:e])) for s, e in chunks(len(v), 64)])]
        out[key] = {**video_rulers(v, ra, rc, rt, ij), "judge_vid_all_acc": acc((pa == ra) & (pc == rc) & (pt == rt))}
    out["frame_consistency_valid"] = out["real"]["frame_acc"] >= VALID_FRAME_ACC
    _calib[name] = (out, real.cpu(), rtv.cpu())                       # 评完挪回 CPU,不占着卡进下一段训练
    del real, rtv
    return _calib[name]


def mismatch_perm(animal, voice, seed):
    """打乱录音用的排列:每一题换成另一种动物的录音。同一音色里各动物条数一样时,按该音色内动物的一个随机错排整组对调
    (音色分布不变);否则在全部题里随机打乱,再把仍同动物的位置两两换开。返回 (perm, 说明);断言每题动物都换了。"""
    rng = np.random.default_rng(seed)
    n, perm, same_voice = len(animal), np.arange(len(animal)), True
    top = max(np.bincount(animal)) if n else 0
    if 2 * top > n:
        raise ValueError(f"某种动物占了 {top}/{n} 题,超过一半,做不到每题都换成别种动物")
    kinds = sorted(set(animal.tolist()))
    for v in sorted(set(voice.tolist())):
        idx = np.nonzero(voice == v)[0]
        groups = {k: idx[animal[idx] == k] for k in kinds}
        if len({len(g) for g in groups.values()}) != 1 or len(kinds) < 2:
            same_voice = False
            break
        while True:                                   # 动物的随机错排:没有哪种动物换回自己
            sigma = dict(zip(kinds, rng.permutation(kinds).tolist()))
            if all(sigma[k] != k for k in kinds):
                break
        for k in kinds:
            perm[groups[k]] = rng.permutation(groups[sigma[k]])
    if not same_voice:
        perm = rng.permutation(n)
        for _ in range(100 * n):
            bad = np.nonzero(animal[perm] == animal)[0]
            if not len(bad):
                break
            i, j = bad[0], rng.integers(n)
            if animal[perm[j]] != animal[i] and animal[perm[i]] != animal[j]:
                perm[i], perm[j] = perm[j], perm[i]
    assert (animal[perm] != animal).all(), "打乱后仍有题的录音是同一种动物"
    return perm, {"seed": seed, "same_voice_kept": bool((voice[perm] == voice).all()), "rule": "按音色整组错排" if same_voice else "全局打乱后换开"}


def evaluate(a, model, data, lays, out):
    """每样本事的对错率 + 导出样例。图和视频交给独立训的小分类器判,念名字交给语音识别 Paraformer 听写(asr.py)。
    起点噪声一律 eval_noise(第 i 题按 eval_seed*10000+i),不用全局随机数。"""
    import frozen
    ij = load_judge(ImageJudge, f"{a.judges}/judge_img.pt", dev)
    vj = load_judge(VideoJudge, f"{a.judges}/judge_vid.pt", dev)
    tasks = set(a.eval_tasks)
    raw_te = np.load(f"{a.data}/world_test.npz")
    raw_voice = np.load(f"{a.data}/voice.npz") if tasks & {"a2t", "a2i"} else None
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
    vmod = lays["t2v"].noisy[0]                                  # vid(Wan 时空潜变量)或 vidimg(逐帧 FLUX.2 潜变量)
    img_vae = frozen.ImageVAE() if tasks & {"t2i", "a2i"} or ("t2v" in tasks and vmod == "vidimg") else None
    vid_vae = frozen.VideoVAE() if "t2v" in tasks and vmod == "vid" else None
    voice = frozen.Voice() if "t2a" in tasks else None

    def gif(frames, path):
        ims = [big(fr, 3) for fr in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0)

    def draw(task, slots, n, cond=None, size=128, first=0):
        x0 = eval_noise(lays[task], n, a.eval_seed, first)
        outs = []
        for s, e in chunks(n, size):
            x = generate(model, lays, task, {k: val[s:e] for k, val in slots.items()}, e - s,
                         cond=[c[s:e].float() for c in cond] if cond else (), cfg=a.cfg, x0=x0[s:e])
            outs.append(x)
        return torch.cat(outs)

    def images(x):
        return torch.cat([img_vae.decode(data.raw("img", x[s:e])) for s, e in chunks(len(x), 128)])

    def videos(z):                                              # 潜变量(未归一化)-> [B, 16, 64, 64, 3]
        if vmod == "vid":
            return vid_vae.decode(z)
        return img_vae.decode(z.flatten(0, 1)).view(len(z), 16, 64, 64, 3)

    if "t2i" in tasks:                            # 1 文字→图:6 动物 x 8 颜色 x 8 张
        ai = torch.arange(6, device=dev).repeat_interleave(64)
        ci = torch.arange(8, device=dev).repeat_interleave(8).repeat(6)
        imgs = images(draw("t2i", {"C": ids(COLOR_ID, ci), "A": ids(ANIMAL_ID, ai)}, len(ai)))
        pa, pc = ij(to_float(imgs))
        pa, pc = pa.argmax(1), pc.argmax(1)
        ok = (pa == ai) & (pc == ci)
        rep["t2i"] = {"n": len(ai), "animal_acc": acc(pa == ai), "color_acc": acc(pc == ci), "both_acc": acc(ok),
                      "both_by_combo": [[acc(ok[(ai == an) & (ci == c)]) for c in range(8)] for an in range(6)]}  # 行=动物、列=颜色,各 8 张
        grid = imgs.view(6, 8, 8, 64, 64, 3)[:, :, 0].permute(0, 2, 1, 3, 4).reshape(6 * 64, 8 * 64, 3)
        Image.fromarray(grid.cpu().numpy()).resize((8 * 128, 6 * 128), Image.NEAREST).save(f"{ex}/gen_t2i_grid.png")
        for k, (an, cn) in enumerate([(0, 0), (1, 4), (2, 6), (3, 2), (4, 5)]):
            big(imgs[an * 64 + cn * 8 + k]).save(f"{ex}/gen_t2i_{k}_{COLOR_ZH[COLORS[cn]]}{ANIMAL_ZH[ANIMALS[an]]}.png")
        # 最近邻:只搜这次实际用的训练图(--per_combo / --hold 的子集)
        train_u8 = torch.from_numpy(np.load(f"{a.data}/world_train.npz")["img"])[data.img.cpu()]
        rep["t2i"]["nearest"] = nearest(imgs, ai, ci, train_u8, raw_te, ex, "t2i")
        if a.hold:                                # 训练时去掉(或对照组照留)的组合:每个另生成 40 张
            hc, ha = [torch.tensor(x, device=dev).repeat_interleave(40) for x in zip(*combos(a.hold))]
            himgs = images(draw("t2i", {"C": ids(COLOR_ID, hc), "A": ids(ANIMAL_ID, ha)}, len(ha), first=1000))
            hpa, hpc = [o.argmax(1) for o in ij(to_float(himgs))]
            hok = (hpa == ha) & (hpc == hc)
            held = torch.zeros(6, 8, dtype=torch.bool, device=dev)
            held[ha, hc] = True
            names = [f"{COLORS[int(hc[k])]}-{ANIMALS[int(ha[k])]}" for k in range(0, len(ha), 40)]
            seen = ~held[ai, ci]
            rep["t2i"]["hold"] = {"combos": names, "match_total": a.match_total, "n_per_combo": 40, "both_acc": acc(hok),
                                  "animal_acc": acc(hpa == ha), "color_acc": acc(hpc == hc),
                                  "by_combo": {nm: {"both_acc": acc(hok[k * 40:(k + 1) * 40]), "animal_acc": acc((hpa == ha)[k * 40:(k + 1) * 40]),
                                                    "color_acc": acc((hpc == hc)[k * 40:(k + 1) * 40])} for k, nm in enumerate(names)},
                                  "seen_grid": {"n": int(seen.sum()), "both_acc": acc(ok[seen]), "animal_acc": acc((pa == ai)[seen]),
                                                "color_acc": acc((pc == ci)[seen])},
                                  "nearest": nearest(himgs, ha, hc, train_u8, raw_te, ex, "hold")}
            for k in range(len(names)):
                big(himgs[k * 40]).save(f"{ex}/gen_hold_{k}_{COLOR_ZH[COLORS[int(hc[k * 40])]]}{ANIMAL_ZH[ANIMALS[int(ha[k * 40])]]}.png")
        del train_u8

    if "t2a" in tasks:                            # 2 文字→念名字:6 x 40 条
        ai = torch.arange(6, device=dev).repeat_interleave(40)
        x = draw("t2a", {"A": ids(ANIMAL_ID, ai)}, len(ai))
        wav = torch.cat([voice.decode(data.raw("aud", x[s:e])) for s, e in chunks(len(x), 64)])
        from asr import Paraformer
        judge = Paraformer(dev)
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
        vids = torch.cat([videos(data.raw(vmod, x[s:e])) for s, e in chunks(len(x), 32)])
        pa, pc, pt = [torch.cat(o).argmax(1) for o in zip(*[vj(to_float(vids[s:e])) for s, e in chunks(len(vids), 64)])]
        # 逐帧一致率旁边一律附同一判卷器在真动画、往返片上的数与有效标记(真片逐帧正确率不到 VALID_FRAME_ACC 即无效,照出不删)
        if vmod == "vid":
            codec, cname = (vid_vae.encode, vid_vae.decode), "wan2.1-vae"
        else:
            codec, cname = (lambda u8: img_vae.encode(u8.flatten(0, 1)).view(len(u8), 16, 32, 8, 8),
                            lambda z: img_vae.decode(z.flatten(0, 1)).view(len(z), 16, 64, 64, 3)), "flux2-vae-per-frame"
        cal, real, rtv = video_calibration(raw_te, ij, vj, codec, cname)
        rep["t2v"] = {"n": len(ai), "animal_acc": acc(pa == ai), "color_acc": acc(pc == ci), "action_acc": acc(pt == ti),
                      "all_acc": acc((pa == ai) & (pc == ci) & (pt == ti)), **video_rulers(vids, ai, ci, ti, ij),
                      "frame_consistency_valid": cal["frame_consistency_valid"], "judge_check": cal}
        for k, j in enumerate([0, 41, 82, 123, 164]):          # 颜色、动物、动作都错开
            gif(vids[j], f"{ex}/gen_t2v_{k}_{COLOR_ZH[COLORS[int(ci[j])]]}{ANIMAL_ZH[ANIMALS[int(ai[j])]]}{ACTION_ZH[ACTIONS[int(ti[j])]]}.gif")
        os.makedirs(f"{ex}/gen_t2v_0_frames", exist_ok=True)   # 第 0 道题(红色 小猫 向左跑)原尺寸逐帧,另存 npy 供看视频的模型原样读
        np.save(f"{ex}/gen_t2v_0_frames/frames.npy", vids[0].cpu().numpy())
        for f in range(16):
            Image.fromarray(vids[0, f].cpu().numpy()).save(f"{ex}/gen_t2v_0_frames/frame_{f:02d}.png")
        for k in range(3):
            gif(real[k], f"{ex}/real_t2v_{k}.gif")
            gif(rtv[k], f"{ex}/roundtrip_t2v_{k}.gif")

    if "i2t" in tasks:                            # 4a 看图说话:测试集 2000 张(种子与训练不重叠)
        n = len(te["img_animal"])
        field = data.key[[x[0] for x in TEMPLATES["i2t"] if isinstance(x, tuple)][0]]   # SigLIP2 特征,或 --see_from img 时 FLUX.2 潜变量
        ans = torch.cat([describe(model, lays, "i2t", [te[field][s:e].to(dev)], e - s) for s, e in chunks(n, 500)])
        c_ok, a_ok = ans[:, 0] == ids(COLOR_ID, te["img_color"]), ans[:, 1] == ids(ANIMAL_ID, te["img_animal"])
        rep["i2t"] = {"n": n, "color_acc": acc(c_ok), "animal_acc": acc(a_ok), "both_acc": acc(c_ok & a_ok)}
        for k in range(5):
            big(raw_te["img"][k]).save(f"{ex}/i2t_{k}_模型说{word(ans[k, 0])}{word(ans[k, 1])}.png")

    if "a2t" in tasks:                            # 4b 听声说名字:没听过的音色,另按测试音色拆开
        ans = describe(model, lays, "a2t", [v[data.key["hear"]][vte.cpu()].to(dev)], len(vte))
        ok = ans[:, 0] == ids(ANIMAL_ID, yte)
        who = raw_voice["voice"][v["wav_index"][vte].cpu().numpy()]
        rep["a2t"] = {"n": len(vte), "animal_acc": acc(ok),
                      "by_voice": {str(w): {"n": int((who == w).sum()), "animal_acc": acc(ok[torch.from_numpy(who == w).to(dev)])}
                                   for w in sorted(set(who.tolist()))}}
        for k in range(5):
            j = k * len(vte) // 5
            sf.write(f"{ex}/a2t_{k}_模型说{word(ans[j, 0])}.wav", raw_voice["wav"][int(v["wav_index"][vte[j]])], SR)

    if "v2t" in tasks:                            # 4c 看视频说出颜色、动物和动作(--watch_frames 时只看取到的那几帧)
        n = len(te["vid_animal"])
        ans = torch.cat([describe(model, lays, "v2t", [te[data.key["watch"]][s:e].to(dev)], e - s) for s, e in chunks(n, 250)])
        c_ok = ans[:, 0] == ids(COLOR_ID, te["vid_color"])
        a_ok = ans[:, 1] == ids(ANIMAL_ID, te["vid_animal"])
        t_ok = ans[:, 2] == ids(ACTION_ID, te["vid_action"])
        rep["v2t"] = {"n": n, "color_acc": acc(c_ok), "animal_acc": acc(a_ok), "action_acc": acc(t_ok), "all_acc": acc(c_ok & a_ok & t_ok)}
        for k in range(3):
            gif(raw_te["vid"][k], f"{ex}/v2t_{k}_模型说{word(ans[k, 0])}{word(ans[k, 1])}{word(ans[k, 2])}.gif")

    if "a2i" in tasks:                            # 5 听到名字画出来:没听过的音色;只判动物
        hear = v[data.key["hear"]][vte.cpu()].to(dev)
        by = lambda ok: {ANIMAL_ZH[ANIMALS[k]]: acc(ok[yte == k]) for k in range(6)}  # noqa: E731

        def listen(cond):
            imgs = images(draw("a2i", {}, len(vte), cond=[cond]))
            return imgs, ij(to_float(imgs))[0].argmax(1)
        imgs, pa = listen(hear)
        rep["a2i"] = {"n": len(vte), "animal_acc": acc(pa == yte), "by_animal": by(pa == yte)}
        for k in range(5):
            j = k * len(vte) // 5
            name = ANIMAL_ZH[ANIMALS[int(yte[j])]]
            sf.write(f"{ex}/in_a2i_{k}_听{name}.wav", raw_voice["wav"][int(v["wav_index"][vte[j]])], SR)
            big(imgs[j]).save(f"{ex}/gen_a2i_{k}_听{name}_画.png")
        if a.a2i_shuffle:                         # 对照:同模型、同噪声、标签不动,录音换成另一种动物的(同一音色内错排,排列按评测种子固定)
            who = raw_voice["voice"][v["wav_index"][vte].cpu().numpy()]
            perm, how = mismatch_perm(yte.cpu().numpy(), who, a.eval_seed)
            perm = torch.from_numpy(perm).to(dev)
            simgs, sa = listen(hear[perm])
            played = yte[perm]                    # 实际放给它听的那条录音是什么动物(每题都和原题不同)
            by_played = lambda ok: {ANIMAL_ZH[ANIMALS[k]]: acc(ok[played == k]) for k in range(6)}  # noqa: E731
            rep["a2i_shuffled"] = {"n": len(vte), "perm": how, "same_animal_after_shuffle": acc(played == yte),
                                   # 按原题的动物算:真在听的麻雀这里该接近 0(每题放的都是别种动物)
                                   "vs_question": {"animal_acc": acc(sa == yte), "by_animal": by(sa == yte)},
                                   # 按实际放的那条录音的动物算:真在听的麻雀这里该和正确录音那组差不多
                                   "vs_played": {"animal_acc": acc(sa == played), "by_animal": by_played(sa == played)}}
            for k in range(5):
                j = k * len(vte) // 5
                name, heard = ANIMAL_ZH[ANIMALS[int(yte[j])]], ANIMAL_ZH[ANIMALS[int(yte[perm[j]])]]
                sf.write(f"{ex}/in_a2i_shuffled_{k}_题{name}_放{heard}.wav", raw_voice["wav"][int(v["wav_index"][vte[perm[j]]])], SR)
                big(simgs[j]).save(f"{ex}/gen_a2i_shuffled_{k}_题{name}_放{heard}_画.png")
    return rep


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


WEIGHTS = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")
MARKERS = ("report.json", "summary.json", "init.json", "records.json", "samples")


def run_products(d):
    """目录里的运行产物:任何权重文件(不论叫什么)、运行标记(报告、init.json、样例目录)与中途评测目录 step<N>/。"""
    try:
        names = os.listdir(d)
    except OSError:
        return []
    return sorted(x for x in names if x.endswith(WEIGHTS) or x in MARKERS or (x.startswith("step") and x[4:].isdigit()))


def fresh_out(out):
    """--out 必须是新目录:里面已有运行产物,或它落在某个运行目录里面(如 runs/x/step1000),都不写。"""
    out = os.path.realpath(out)
    taken = run_products(out)
    if taken:
        raise SystemExit(f"{out} 里已有 {taken};不覆盖,换一个 --out(要重跑先删掉它)")
    up = os.path.dirname(out)
    while up != os.path.dirname(up):
        if any(x.endswith(WEIGHTS) or x in MARKERS[:3] for x in run_products(up)):
            raise SystemExit(f"{out} 在运行目录 {up} 里面;不往别的运行目录里写")
        up = os.path.dirname(up)
    os.makedirs(out, exist_ok=True)


def write_json(obj, path):
    """原子写:先写临时文件再改名,读的一方不会读到半个文件。"""
    tmp = f"{path}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def identity(a, model, built, start):
    """init.json 的内容:这次尝试的身份(排队脚本给的 SPARROW_ATTEMPT)、训练随机协议与种子、随机构造时的初始化哈希
    (built,全部参数与逐个参数)、实际训练起点的哈希(start:从零就是构造的,--init / --eval_only 是加载权重之后的)。"""
    return {"attempt": os.environ.get("SPARROW_ATTEMPT", ""), "rng": a.rng, "seed": a.seed,
            "built_sha256_all": params_sha256(built), "trunk_init_sha256": params_sha256(built, TRUNK),
            "param_sha256": {k: [list(v.shape), params_sha256({k: v})] for k, v in built.items()},
            "start_sha256_all": params_sha256(model.state_dict()), "start": start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--out", required=True, help="新目录;已有报告、样例、权重的不写")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--eval_tasks", default="", help="评测哪些本事;不写就是 --tasks(--eval_only 时是那份权重训练时评测的那几样)")
    ap.add_argument("--init", help="从这份 final.pt 的权重接着训(上一集);dim/depth/heads/aud_split/maxlen 须一致;归一化统计量沿用它的")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--stop_at", type=int, default=0, help="学习率照 --steps 排,练到第这么多步就停下存权重、评测(原样重走某次运行的前一段)")
    ap.add_argument("--seed", type=int, default=0, help="训练种子")
    ap.add_argument("--rng", default="legacy", choices=["legacy", "paired"],
                    help="随机协议:legacy 全部吃全局随机数(九月的权重就是这样练的);paired 初始化按部件、训练按任务与用途各用独立随机流,"
                         "同种子的两版对照公共主干初始化相同,改一个任务的批量不改别的任务抽到的样本")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--maxlen", type=int, default=400, help="位置表多长;比较不同序列长度的两版时两版写同一个值")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--bs_vid", type=int, default=128)
    ap.add_argument("--bs_task", default="", help="单独给某些任务每步的样本数(正整数),如 t2i=51;没写的用 --bs / --bs_vid")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--eval_seed", type=int, default=0, help="评测噪声种子:各任务第 i 题起点噪声按 eval_seed*10000+i 生成")
    ap.add_argument("--aud_split", type=int, default=4, help="每个 Mimi 帧(512 维)拆成几个向量进主干")
    ap.add_argument("--vid_attn", default="full", choices=["full", "frame"], help="视频段内:各帧互相看得见 / 每帧只看自己")
    ap.add_argument("--vid_lat", default="vid", choices=["vid", "vidimg"], help="生视频用 Wan 时空潜变量 / 逐帧 FLUX.2 潜变量")
    ap.add_argument("--see_from", default="see", choices=["see", "img"], help="看图用 SigLIP2 特征 / FLUX.2 潜变量(都经两层投影)")
    ap.add_argument("--watch_frames", type=int, default=16, help="看视频每段取几帧(16 帧均分成几段、各取段中点那帧)")
    ap.add_argument("--per_combo", type=int, default=0, help="每种颜色x动物只用前 K 张训练图(0 = 全部)")
    ap.add_argument("--hold", default="", help="训练图去掉这些颜色-动物组合,如 red-rabbit,blue-cat;评测时这些组合各另生成 40 张")
    ap.add_argument("--match_total", action="store_true", help="和 --hold 一起用:那些组合照留,改为随机去掉同样多张(对照)")
    ap.add_argument("--voices", type=int, default=0, help="训练录音只用几个音色(0 = 全部)")
    ap.add_argument("--voice_draw", type=int, default=0, help="--voices 的抽法编号:挑哪几个音色、各取哪几条录音")
    ap.add_argument("--clips", type=int, default=36, help="--voices 时一共用几条训练录音(每种动物一样多)")
    ap.add_argument("--weights", default="t2a=4", help="各任务损失的权重;没写的都是 1")
    ap.add_argument("--eval_every", type=int, default=0, help="每训这么多步评测一次(0=只在最后评测)")
    ap.add_argument("--eval_at", default="", help="另在这些步评测,逗号隔开,如 25,50,75(和 --eval_every 取并集)")
    ap.add_argument("--keep_midway", action="store_true", help="中途评测时另存当时的权重 step<步数>/final.pt")
    ap.add_argument("--a2i_shuffle", action="store_true", help="评测听→画时另做一遍录音换成别种动物(标签与噪声不动)的对照")
    ap.add_argument("--eval_only", action="store_true", help="不训练,只评测 --ckpt;报告与样例写进 --out")
    ap.add_argument("--ckpt", help="--eval_only 评测的那份 final.pt")
    a = ap.parse_args()
    a.tasks = a.tasks.split(",")
    a.weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    a.bs_task = {k: int(v) for k, v in (x.split("=") for x in a.bs_task.split(",") if x)}
    if any(v <= 0 for v in a.bs_task.values()):
        raise SystemExit(f"--bs_task 要正整数:{a.bs_task};不练某个任务就别写进 --tasks")
    ck = None
    if a.eval_only:
        if not a.ckpt:
            raise SystemExit("--eval_only 要用 --ckpt 指定评测哪份权重")
        ck = torch.load(a.ckpt, map_location=dev)
        trained = with_defaults(ck["args"])
        for k in DEFAULTS:                        # 结构与训练子集照那份权重
            setattr(a, k, trained[k])
        a.rng, a.seed = trained.get("rng", "legacy"), trained.get("seed", 0)   # 训练协议照那份权重(九月的旧权重:旧协议、种子 0)
        a.eval_tasks = a.eval_tasks.split(",") if a.eval_tasks else trained.get("eval_tasks") or trained["tasks"]
        a.tasks, a.steps, a.init = [], 0, None
    else:
        a.eval_tasks = a.eval_tasks.split(",") if a.eval_tasks else a.tasks
    a.eval_points = sorted({int(x) for x in a.eval_at.split(",") if x} |
                           (set(range(a.eval_every, a.steps, a.eval_every)) if a.eval_every else set()))
    fresh_out(a.out)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure(vars(a))
    t0 = time.time()
    parent = ck or (torch.load(a.init, map_location=dev) if a.init else None)
    inherited = run_mods(with_defaults(parent["args"])) if a.init else ()      # 上一集本来就用的模态:统计量必须在它的权重里
    data = Cache(a.cache, needs(a.tasks + a.eval_tasks), vars(a), np.load(f"{a.data}/voice.npz") if a.voices else None,
                 parent.get("stat") if parent else None, strict=a.eval_only, inherited=inherited)
    lays = {t: Layout(t, a.vid_attn == "frame") for t in TASKS}
    long = {t: lays[t].L for t in a.tasks + a.eval_tasks if lays[t].L > a.maxlen}
    if long:
        raise SystemExit(f"序列比 --maxlen {a.maxlen} 长:{long}")
    model = Sparrow(a.dim, a.depth, a.heads, a.maxlen, seed=a.seed if a.rng == "paired" else None).to(dev)
    built = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rep = {"args": vars(a)}
    if parent:
        pa = with_defaults(parent["args"])
        bad = {k: (pa[k], vars(a)[k]) for k in ("dim", "depth", "heads", "aud_split", "maxlen") if pa[k] != vars(a)[k]}
        if bad:
            raise SystemExit(f"--init {a.init} 的结构和这次不一致(上一集, 这次):{bad}")
        missing, unexpected = model.load_state_dict(parent["model"], strict=False)
        fresh = [k for k in missing if k.split(".")[1] in ("seelat", "vidimg")]      # 上一集没有的可选表示,新初始化
        if unexpected or len(fresh) != len(missing):
            raise SystemExit(f"--init 的权重对不上:缺 {sorted(set(missing) - set(fresh))},多 {unexpected}")
        if fresh:
            rep["init_fresh_params"] = fresh
        if a.init:
            rep["init_sha256"] = sha256(a.init)
    start = f"eval_only:{sha256(a.ckpt)}" if a.eval_only else (f"init:{rep['init_sha256']}" if a.init else "random_init")
    rep["init"] = identity(a, model, built, start)
    if a.eval_only:
        rep["init"]["eval_protocol"] = {"eval_seed": a.eval_seed, "cfg": a.cfg, "noise": "各任务第 i 题起点噪声种子 = eval_seed*10000+i"}
    write_json(rep["init"], f"{a.out}/init.json")                     # 起跑就写(原子):成对的几版马上核得了
    used = sorted(needs(a.tasks + a.eval_tasks))
    files = [f"{a.data}/{f}" for f in ("world_train.npz", "world_test.npz", "voice.npz")] + [f"{a.cache}/labels.npz"]
    rep.update({"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "proj_params": {m: sum(p.numel() for p in model.inp[m].parameters()) for m in used},
                "seq_len": {t: lays[t].L for t in TASKS}, "batch_per_step": {t: batch(a, t) for t in a.tasks},
                "exposure": {t: batch(a, t) * (a.stop_at or a.steps) for t in a.tasks}, "train_subset": data.subset,
                "train_data": {"images": len(data.img), "videos": len(data.tr["vid_animal"]), "voice_rows": len(data.vtr)},
                "data_sha256": {os.path.basename(f): sha256(f) for f in files},
                "stat_source": data.stat_source, "load_seconds": round(time.time() - t0, 1)})
    if a.eval_only:
        rep["eval_only"] = {"ckpt": a.ckpt, "ckpt_sha256": sha256(a.ckpt), "ckpt_args": ck["args"]}
    else:
        reset_peak()
        rep["train_seconds"], rep["log"], rep["train_peak_mem_GB"], rep["midway_eval_peak_mem_GB"] = train(a, model, data, lays)
        save(model, a, data, f"{a.out}/final.pt")
    model.eval()
    t0 = time.time()
    reset_peak()
    rep["eval_config"] = {"cfg": a.cfg, "sample_steps": 30, "eval_seed": a.eval_seed, "tasks": a.eval_tasks,
                          "noise": "各任务第 i 题起点噪声种子 = eval_seed*10000+i(九月的旧报告用全局种子 1234 的随机数,两者不能拼在一条曲线上)"}
    rep["eval"] = evaluate(a, model, data, lays, a.out)
    rep["eval_seconds"] = round(time.time() - t0, 1)
    rep["eval_peak_mem_GB"] = round(peak(), 2)
    write_json(rep, f"{a.out}/report.json")
    print(json.dumps({k: val for k, val in rep.items() if k != "log"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

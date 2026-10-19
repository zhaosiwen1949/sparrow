"""拿训好的 final.pt 直接用:不要缓存,只要权重和用得到的那个冻结零件。用哪个零件、看视频取哪几帧、视频怎么解码,都照这份权重训练时的配置。

  python infer.py O/final.pt t2i "红色 小猫" 猫.png
  python infer.py O/final.pt t2a 小猫 猫.wav
  python infer.py O/final.pt t2v "蓝色 小狗 向右跑" 狗.gif      # 另存 狗_frames/frame_00..15.png 与 frames.npy
  python infer.py O/final.pt i2t 图.png               # 64x64 的图 -> 说出颜色、动物
  python infer.py O/final.pt a2t 录音.wav             # 去静音后 2 秒以内 -> 说出动物
  python infer.py O/final.pt v2t 动画.gif             # 16 帧 64x64(.gif、16 帧的 .npy 或放 16 张 png 的目录)-> 说出颜色、动物、动作
  python infer.py O/final.pt a2i 录音.wav 画.png      # 听到名字画出来
生成默认从种子 0 的那团噪声出发(S.start_noise),30 步、CFG 3;--seed N 换一团,--noise 文件.pt 用存好的那团(形状 [1, 向量数, 维度])。
生成的同时存下起点噪声(<输出>.noise.pt)。--nearest K --data D:画出来的图另找这份权重训练时实际用过的训练图里最像的 K 张
(每个像素通道的均方根误差,同 sparrow.py 的最近邻)。--record 文件:每次调用追加一行 JSON(权重哈希、输入、噪声、回答或输出、近邻)。
只能用那一集训练或评测过的本事(final.pt 只存了用到的那几种向量的归一化统计量);缺的要用 --stat_from 另一份权重明说从哪儿取。
"""
import argparse
import hashlib
import json
import os

import numpy as np
import soundfile as sf
import torch
from PIL import Image, ImageSequence

import frozen
import sparrow as S
from parts import ID, SR, VOCAB, fit_clip

dev = S.dev


def to64(im):
    """读进来的图缩到 64x64:正好是 64 的整数倍的(本仓库导出的放大图)按最近邻缩回,逐像素还原;其余双线性。"""
    im = im.convert("RGB")
    exact = im.size[0] % 64 == 0 and im.size == (im.size[0], im.size[0])
    return np.asarray(im.resize((64, 64), Image.NEAREST if exact else Image.BILINEAR))


def picture(path):
    return torch.from_numpy(to64(Image.open(path)).copy())[None].to(dev)


def movie(path):
    """16 帧 [1, 16, 64, 64, 3]:.npy、放 16 张 png 的目录、或 gif。"""
    if path.endswith(".npy"):
        frames = np.load(path)
    elif os.path.isdir(path):
        frames = np.stack([to64(Image.open(f"{path}/{f}")) for f in sorted(os.listdir(path)) if f.endswith(".png")])
    else:
        frames = np.stack([to64(f) for f in ImageSequence.Iterator(Image.open(path))])
    if len(frames) != 16:
        raise SystemExit(f"{path} 有 {len(frames)} 帧,要 16 帧")
    return torch.from_numpy(np.ascontiguousarray(frames))[None].to(dev)


def recording(path):
    import librosa
    clip, sec = fit_clip(librosa.load(path, sr=SR)[0])
    if clip is None:
        raise SystemExit(f"{path} 去静音后 {sec:.2f} 秒,装不下 2.08 秒")
    return torch.from_numpy(clip)[None].to(dev)


def tensor_sha256(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("task", choices=S.TASKS)
    ap.add_argument("input", help="t2i/t2a/t2v:空格隔开的词;其余:图/录音/动画文件")
    ap.add_argument("out", nargs="?", help="生成的本事写到这个文件")
    ap.add_argument("--seed", type=int, default=0, help="起点噪声种子(S.start_noise)")
    ap.add_argument("--noise", help="起点噪声文件(torch.save 的张量),给了就不用 --seed")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--stat_from", help="这份权重缺某种向量的归一化统计量时,从另一份权重取(只补缺的,来源记进 --record)")
    ap.add_argument("--nearest", type=int, default=0, help="t2i / a2i:另找最像的 K 张训练图")
    ap.add_argument("--data", help="--nearest 要用的数据目录(world_train.npz 所在)")
    ap.add_argument("--record", help="把这次调用追加成一行 JSON 写进这个文件")
    a = ap.parse_args()
    model, lays, stat = S.load(a.ckpt)
    ck_args = S.with_defaults(torch.load(a.ckpt, map_location="cpu")["args"])
    lay = lays[a.task]
    rec = {"ckpt": a.ckpt, "ckpt_sha256": S.sha256(a.ckpt), "task": a.task, "input": a.input, "out": a.out}
    mods = [x[0] for x in S.TEMPLATES[a.task] if isinstance(x, tuple)]
    if a.stat_from:
        other = {m: [x.to(dev) for x in st] for m, st in torch.load(a.stat_from, map_location="cpu")["stat"].items()}
        filled = [m for m in mods if m not in stat and m in other]
        stat.update({m: other[m] for m in filled})
        rec["stat_from"] = {"ckpt": a.stat_from, "ckpt_sha256": S.sha256(a.stat_from), "mods": filled}
    missing = [m for m in mods if m not in stat]
    if missing:
        raise SystemExit(f"这份权重没有 {missing} 的归一化统计量(那一集没训也没评这样本事);要用就加 --stat_from 另一份权重")
    big = lambda u8: Image.fromarray(u8.cpu().numpy()).resize((256, 256), Image.NEAREST)  # noqa: E731

    cond, slots = [], {}
    if a.task in ("t2i", "t2a", "t2v"):
        words = a.input.split()
        if len(words) != len(lay.slots) or any(w not in ID for w in words):
            raise SystemExit(f"{a.task} 要 {len(lay.slots)} 个词({'/'.join(lay.slots)}),词表:{' '.join(VOCAB[:18])}")
        slots = {s: torch.tensor([ID[w]], device=dev) for s, w in zip(lay.slots, words)}
    else:
        with torch.autocast("cuda", torch.bfloat16):                    # 和 encode.py 编缓存时同一精度
            if a.task == "i2t" and mods[0] == "seelat":                 # 看图用 FLUX.2 潜变量的权重
                cond = [frozen.ImageVAE().encode(picture(a.input))]
            elif a.task == "i2t":
                cond = [frozen.See().encode(picture(a.input), grid=4)]
            elif a.task == "v2t":                                       # 每帧 2x2 个 SigLIP2 向量,再按这份权重的取帧规则留几帧
                f = frozen.See().encode(movie(a.input)[0], grid=2)       # [16, 4, 768]
                cond = [f[S.watch_index(ck_args["watch_frames"])].reshape(1, -1, f.shape[-1])]
                rec["watch_frames"] = S.watch_index(ck_args["watch_frames"])
            else:
                cond = [frozen.Hear().encode(recording(a.input))]
    cond = [S.norm(stat, m, c.float()) for m, c in zip(mods, cond)]

    if a.task in ("i2t", "a2t", "v2t"):
        words = [VOCAB[int(i)] for i in S.describe(model, lays, a.task, cond, 1)[0]]
        rec["answer"] = words
        print(" ".join(words))
    else:
        x0 = torch.load(a.noise, map_location=dev).float().reshape(1, -1, S.MODS[lay.noisy[0]][0]) if a.noise else S.start_noise(lay, 1, a.seed)
        rec.update({"noise": {"file": a.noise, "tensor_sha256": tensor_sha256(x0)} if a.noise else {"seed": a.seed},
                    "cfg": a.cfg, "steps": a.steps})
        x = S.generate(model, lays, a.task, slots, 1, cond=cond, cfg=a.cfg, steps=a.steps, x0=x0)
        mod = lay.noisy[0]
        if mod == "vidimg":                                             # 逐帧 FLUX.2 潜变量:每帧各自解码
            y = frozen.ImageVAE().decode(S.denorm(stat, mod, x).flatten(0, 1))
        else:
            y = {"img": frozen.ImageVAE, "aud": frozen.Voice, "vid": frozen.VideoVAE}[mod]().decode(S.denorm(stat, mod, x))[0]
        if not a.out:
            raise SystemExit("生成的本事要给输出文件")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save(x0.cpu(), f"{a.out}.noise.pt")
        if mod == "img":
            big(y).save(a.out)
        elif mod == "aud":
            sf.write(a.out, y.cpu().numpy(), SR)
        else:
            ims = [big(f) for f in y]
            ims[0].save(a.out, save_all=True, append_images=ims[1:], duration=100, loop=0)
            fd = f"{os.path.splitext(a.out)[0]}_frames"
            os.makedirs(fd, exist_ok=True)
            np.save(f"{fd}/frames.npy", y.cpu().numpy())
            for k, f in enumerate(y):
                Image.fromarray(f.cpu().numpy()).save(f"{fd}/frame_{k:02d}.png")
        if a.nearest and mod == "img":
            raw = np.load(f"{a.data}/world_train.npz")
            pick = S.pick_images(raw["img_color"], raw["img_animal"], ck_args["per_combo"], ck_args["hold"], ck_args["match_total"])
            train = torch.from_numpy(raw["img"])[pick]
            d, i = S.topk_rmse(y[None], train, a.nearest)
            stem = os.path.splitext(a.out)[0]
            for k in range(a.nearest):
                big(train[int(i[0, k])]).save(f"{stem}_nn{k}.png")
            rec["nearest"] = {"metric": "每个像素通道的均方根误差(0~255)", "searched": len(train),
                              "train_index": [int(pick[int(j)]) for j in i[0]], "rmse": [round(float(x), 2) for x in d[0]]}
        print(a.out)
    if a.record:
        with open(a.record, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""拿训好的 final.pt 直接用:不要数据、不要缓存,只要权重和用得到的那个冻结零件。

  python infer.py O/final.pt t2i "红色 小猫" 猫.png
  python infer.py O/final.pt t2a 小猫 猫.wav
  python infer.py O/final.pt t2v "蓝色 小狗 向右跑" 狗.gif
  python infer.py O/final.pt i2t 图.png               # 64x64 的图 -> 说出颜色、动物
  python infer.py O/final.pt a2t 录音.wav             # 去静音后 2 秒以内 -> 说出动物
  python infer.py O/final.pt v2t 动画.gif             # 16 帧 64x64 -> 说出颜色、动物、动作
  python infer.py O/final.pt a2i 录音.wav 画.png      # 听到名字画出来
生成从种子 0 的那团噪声出发(S.start_noise),30 步、CFG 3;同提示每次结果一样,和 ep3_assets.py 的「一步步生成」末帧一致。
只能用那一集训练或评测过的本事(final.pt 里只存了用到的那几种向量的归一化统计量)。
"""
import argparse

import numpy as np
import soundfile as sf
import torch
from PIL import Image, ImageSequence

import frozen
import sparrow as S
from parts import ID, SR, VOCAB, fit_clip

dev = "cuda"


def picture(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB").resize((64, 64), Image.BILINEAR)))[None].to(dev)


def recording(path):
    import librosa
    clip, sec = fit_clip(librosa.load(path, sr=SR)[0])
    if clip is None:
        raise SystemExit(f"{path} 去静音后 {sec:.2f} 秒,装不下 2.08 秒")
    return torch.from_numpy(clip)[None].to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("task", choices=S.TASKS)
    ap.add_argument("input", help="t2i/t2a/t2v:空格隔开的词;其余:图/录音/动画文件")
    ap.add_argument("out", nargs="?", help="生成的本事写到这个文件")
    a = ap.parse_args()
    model, lays, stat = S.load(a.ckpt)
    lay = lays[a.task]
    big = lambda u8: Image.fromarray(u8.cpu().numpy()).resize((256, 256), Image.NEAREST)  # noqa: E731

    cond, slots = [], {}
    if a.task in ("t2i", "t2a", "t2v"):
        words = a.input.split()
        if len(words) != len(lay.slots) or any(w not in ID for w in words):
            raise SystemExit(f"{a.task} 要 {len(lay.slots)} 个词({'/'.join(lay.slots)}),词表:{' '.join(VOCAB[:18])}")
        slots = {s: torch.tensor([ID[w]], device=dev) for s, w in zip(lay.slots, words)}
    else:
        with torch.autocast("cuda", torch.bfloat16):                    # 和 encode.py 编缓存时同一精度
            if a.task == "i2t":
                cond = [frozen.See().encode(picture(a.input), grid=4)]
            elif a.task == "v2t":
                frames = [np.asarray(f.convert("RGB").resize((64, 64))) for f in ImageSequence.Iterator(Image.open(a.input))][:16]
                cond = [frozen.See().encode(torch.from_numpy(np.stack(frames)).to(dev), grid=2).reshape(1, -1, 768)]
            else:
                cond = [frozen.Hear().encode(recording(a.input))]
    mods = [x[0] for x in S.TEMPLATES[a.task] if isinstance(x, tuple)]
    cond = [S.norm(stat, m, c) for m, c in zip(mods, cond)]

    if a.task in ("i2t", "a2t", "v2t"):
        print(" ".join(VOCAB[int(i)] for i in S.describe(model, lays, a.task, cond, 1)[0]))
        return
    x = S.generate(model, lays, a.task, slots, 1, cond=cond, x0=S.start_noise(lay, 1, 0))
    mod = lay.noisy[0]
    y = {"img": frozen.ImageVAE, "aud": frozen.Voice, "vid": frozen.VideoVAE}[mod]().decode(S.denorm(stat, mod, x))[0]
    if mod == "img":
        big(y).save(a.out)
    elif mod == "aud":
        sf.write(a.out, y.cpu().numpy(), SR)
    else:
        ims = [big(f) for f in y]
        ims[0].save(a.out, save_all=True, append_images=ims[1:], duration=100, loop=0)
    print(a.out)


if __name__ == "__main__":
    main()

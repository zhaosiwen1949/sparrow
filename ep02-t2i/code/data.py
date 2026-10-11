"""把卡通小世界与 TTS 录音整理成训练用的 npz。

  python data.py world <out.npz> <图张数> <视频段数> <起始种子>   # 每条样本的种子 = 起始种子 + 序号
  python data.py voice <wav 目录> <out.npz>                      # speak.py 的产物 -> 24 kHz、统一 2.08 秒(只补尾部静音)

测试集用另一段种子(见 run.sh),所以测试图一张都没在训练里出现过。
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "world"))
import world  # noqa: E402

from parts import ACTIONS, ANIMAL_ZH, ANIMALS, COLORS, SR, fit_clip  # noqa: E402

def one_img(seed):
    s = world.sample(np.random.default_rng(seed))
    return s["image"], ANIMALS.index(s["animal"]), COLORS.index(s["color"])


def one_vid(seed):
    rng = np.random.default_rng(seed)
    action = ACTIONS[rng.integers(len(ACTIONS))]
    v = world.animate(rng, action)
    return np.stack(v["frames"]), ANIMALS.index(v["animal"]), COLORS.index(v["color"]), ACTIONS.index(action)


def make_world(out, n_img, n_vid, seed):
    t0 = time.time()
    with Pool(min(96, os.cpu_count())) as pool:
        im = pool.map(one_img, range(seed, seed + n_img), chunksize=32)
        vi = pool.map(one_vid, range(seed + 50_000_000, seed + 50_000_000 + n_vid), chunksize=4)
    np.savez(out, img=np.stack([r[0] for r in im]), img_animal=np.array([r[1] for r in im]),
             img_color=np.array([r[2] for r in im]), vid=np.stack([r[0] for r in vi]),
             vid_animal=np.array([r[1] for r in vi]), vid_color=np.array([r[2] for r in vi]),
             vid_action=np.array([r[3] for r in vi]))
    print(f"world {out}: {n_img} 张图 + {n_vid} 段视频,{time.time() - t0:.0f}s,{os.path.getsize(out) / 1e6:.0f} MB")


def make_voice(src, out):
    """去掉首尾静音,前面留 50 ms,尾部补静音到统一的 2.08 秒;装不下的、合成时没念对的(_fail)整条不要(绝不截掉字)。
    音色名以 te 开头的是测试音色(FLEURS test 份的说话人),其余训练用。"""
    import librosa
    names = sorted(f for f in os.listdir(src) if f.endswith(".wav") and not f.endswith(".part.wav"))
    wavs, animals, voices, secs, kept = [], [], [], [], []
    dropped = [(f, "合成时试满 4 次仍没念对") for f in names if f.endswith("_fail.wav")]
    names = [f for f in names if not f.endswith("_fail.wav")]
    zh = {v: k for k, v in ANIMAL_ZH.items()}
    for f in names:
        word, voice, _, _ = f[:-4].split("_")
        y, _ = librosa.load(os.path.join(src, f), sr=SR)
        clip, speech = fit_clip(y)
        if clip is None:
            dropped.append((f, f"去静音后 {speech:.2f} 秒,装不下"))
            continue
        secs.append(speech)
        wavs.append(clip)
        animals.append(ANIMALS.index(zh[word]))
        voices.append(voice)
        kept.append(f)
    test = np.array([v.startswith("te") for v in voices])
    np.savez(out, wav=np.stack(wavs), animal=np.array(animals), voice=np.array(voices), test=test, name=np.array(kept))
    json.dump(dropped, open(out.replace(".npz", "_rejected.json"), "w"), ensure_ascii=False)       # 把关时按「没念对」算
    print(f"voice {out}: {len(wavs)} 条,测试 {test.sum()} 条,{len(set(voices))} 个音色;"
          f"去静音后长度 {min(secs):.2f}~{max(secs):.2f} 秒;不要的 {len(dropped)} 条 {dropped[:5]}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or len(sys.argv) < (6 if sys.argv[1] == "world" else 4):
        print("用法:python data.py world <out.npz> <图张数> <视频段数> <起始种子>  |  python data.py voice <wav 目录> <out.npz>")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    if sys.argv[1] == "world":
        make_world(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        make_voice(sys.argv[2], sys.argv[3])

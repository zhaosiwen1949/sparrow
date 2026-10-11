"""训练前把数据过一遍冻结零件,缓存到磁盘(fp16);训练时只跑小主干。

按零件分文件,只编用得到的(课程逐集加本事,硬件要求随之递增):
  img   FLUX.2 VAE  图的潜变量(生图、听→画)     vid   Wan VAE   动画的潜变量(生视频)
  aud   Mimi        录音的潜变量(念名字)         see   SigLIP2   看图      watch SigLIP2 看动画(每帧 2x2)
  hear  Whisper     听录音(听声、听→画)
顺便量图和视频的压缩器「压了再还原」还剩多少:还原结果交给判卷分类器,准确率就是这条路的上限。

  python encode.py --data D --judges J --out C [--parts img,aud,...]
产出 C/labels.npz(标签,每次都写)、C/<零件>.npz(键名「份.字段」,份是 train/test/voice)
与 C/encode.json(每个零件的参数量、耗时、显存峰值、还原后的判卷准确率、缓存大小;多次编码合并进同一份)。
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import frozen
from parts import SR, ImageJudge, VideoJudge, load_judge, tail_room, to_float

dev = "cuda"
AUG = 4            # 训练用录音每条做 4 份(原样 + 3 份随机后移 0~100ms,只挪进尾部静音,不截字),测试录音只 1 份
PARTS = ["img", "vid", "aud", "see", "watch", "hear"]


def run(fn, x, bs, **kw):
    """把 numpy/张量 x 分批送上卡过 fn,结果拼回来(留在卡上)。"""
    out = []
    for k in range(0, len(x), bs):
        xb = torch.as_tensor(x[k:k + bs]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(fn(xb, **kw).float())
    return torch.cat(out)


def half(t):
    return t.half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args()
    parts = a.parts.split(",")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    tr, te = np.load(f"{a.data}/world_train.npz"), np.load(f"{a.data}/world_test.npz")
    voice = np.load(f"{a.data}/voice.npz")
    meta = f"{a.out}/encode.json"
    rep = json.load(open(meta)) if os.path.exists(meta) else {}
    acc = lambda p, y: (p.argmax(1).cpu() == torch.as_tensor(y)).float().mean().item()  # noqa: E731

    labels = {f"{s}.{k}": d[k] for s, d in (("train", tr), ("test", te))
              for k in ("img_animal", "img_color", "vid_animal", "vid_color", "vid_action")}
    vtest = voice["test"]
    labels["voice.animal"] = np.concatenate([voice["animal"][~vtest]] * AUG + [voice["animal"][vtest]])
    labels["voice.test"] = np.concatenate([np.zeros((~vtest).sum() * AUG, bool), np.ones(vtest.sum(), bool)])
    labels["voice.wav_index"] = np.concatenate([np.nonzero(~vtest)[0]] * AUG + [np.nonzero(vtest)[0]])
    np.savez(f"{a.out}/labels.npz", **labels)

    def wav():
        """训练录音做 AUG 份,测试录音 1 份(随机数固定,aud 和 hear 拿到的是同一批)。"""
        g = np.random.default_rng(0)
        wtr, wte = voice["wav"][~vtest], voice["wav"][vtest]
        room = np.minimum(tail_room(torch.from_numpy(wtr)).numpy(), SR // 10)
        shifted = [wtr] + [np.stack([np.roll(w, int(g.integers(0, r + 1))) for w, r in zip(wtr, room)]) for _ in range(AUG - 1)]
        return np.concatenate(shifted + [wte])

    def img(m):
        lat = run(m.encode, te["img"], 256)
        rec = run(m.decode, lat[:500], 250).byte()
        pa, pc = ij(to_float(rec))
        return ({"train.img_lat": half(run(m.encode, tr["img"], 256)), "test.img_lat": half(lat)},
                {"latent": list(lat.shape[1:]),
                 "recon_l1": (to_float(rec) - to_float(torch.as_tensor(te["img"][:500]).to(dev))).abs().mean().item(),
                 "recon_animal_acc": acc(pa, te["img_animal"][:500]), "recon_color_acc": acc(pc, te["img_color"][:500])})

    def vid(m):
        lat = run(m.encode, te["vid"], 32)
        rec = run(m.decode, lat[:200], 25).byte()
        pa, pc, pt = vj(to_float(rec))
        return ({"train.vid_lat": half(run(m.encode, tr["vid"], 32)), "test.vid_lat": half(lat)},
                {"latent": list(lat.shape[1:]), "recon_animal_acc": acc(pa, te["vid_animal"][:200]),
                 "recon_color_acc": acc(pc, te["vid_color"][:200]), "recon_action_acc": acc(pt, te["vid_action"][:200])})

    def aud(m):
        lat = half(run(m.encode, wav(), 64))
        return {"voice.aud_lat": lat}, {"latent": list(lat.shape[1:])}

    def see(m):
        out = {f"{s}.img_see": half(run(m.encode, d["img"], 512, grid=4)) for s, d in (("train", tr), ("test", te))}
        return out, {"tokens": list(out["test.img_see"].shape[1:])}

    def watch(m):
        out = {}
        for s, d in (("train", tr), ("test", te)):
            v = d["vid"]
            f = run(m.encode, v.reshape(-1, 64, 64, 3), 512, grid=2)
            out[f"{s}.vid_see"] = half(f.view(len(v), -1, f.shape[-1]))                  # 16 帧 x 4 = 64 个
        return out, {"tokens": list(out["test.vid_see"].shape[1:])}

    def hear(m):
        out = half(run(m.encode, wav(), 128))
        return {"voice.aud_hear": out}, {"tokens": list(out.shape[1:])}

    job = {"img": (frozen.ImageVAE, "image_vae", img), "vid": (frozen.VideoVAE, "video_vae", vid), "aud": (frozen.Voice, "voice", aud),
           "see": (frozen.See, "see", see), "watch": (frozen.See, "see", watch), "hear": (frozen.Hear, "hear", hear)}
    ij = load_judge(ImageJudge, f"{a.judges}/judge_img.pt", dev)
    vj = load_judge(VideoJudge, f"{a.judges}/judge_vid.pt", dev)
    for p in parts:
        cls, repo, fn = job[p]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        m = cls()
        load_s = round(time.time() - t0, 1)
        t0 = time.time()
        arrays, met = fn(m)
        np.savez(f"{a.out}/{p}.npz", **arrays)
        rep[p] = {"repo": "/".join(x for x in frozen.REPOS[repo] if x), "params_M": frozen.params_M(m), "load_seconds": load_s,
                  "encode_seconds": round(time.time() - t0, 1), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  "cache_MB": round(os.path.getsize(f"{a.out}/{p}.npz") / 1e6, 1), **met}
        print(p, json.dumps(rep[p], ensure_ascii=False), flush=True)
        del m, arrays
        torch.cuda.empty_cache()
    json.dump(rep, open(meta, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

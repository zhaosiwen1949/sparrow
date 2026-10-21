"""判卷用的两个小分类器:图(动物+颜色)、视频(动物+颜色+动作)。只用真数据训,和主模型无关。
念名字不用分类器判,用语音识别听写(见 asr.py)。

  python train_judges.py img|vid --data D --out O [--steps N]
D 里要有 world_train.npz、world_test.npz;产出 O/judge_<名字>.pt 与 .json(耗时、显存峰值、测试准确率)。
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from parts import ImageJudge, VideoJudge, to_float

dev = "cuda"


def load(path):
    d = np.load(path)
    return {k: torch.from_numpy(d[k]).to(dev) for k in d.files if d[k].dtype.kind != "U"}  # 跳过字符串字段


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["img", "vid"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    pick = lambda n, bs: torch.randint(0, n, (bs,), device=dev)  # noqa: E731
    tr, te = load(f"{a.data}/world_train.npz"), load(f"{a.data}/world_test.npz")
    if a.what == "img":
        m = ImageJudge().to(dev)

        def loss_fn():
            i = pick(len(tr["img"]), 128)
            pa, pc = m(to_float(tr["img"][i]))
            return F.cross_entropy(pa, tr["img_animal"][i]) + F.cross_entropy(pc, tr["img_color"][i])
    else:
        m = VideoJudge().to(dev)

        def loss_fn():
            i = pick(len(tr["vid"]), 16)
            pa, pc, pt = m(to_float(tr["vid"][i]))
            return (F.cross_entropy(pa, tr["vid_animal"][i]) + F.cross_entropy(pc, tr["vid_color"][i])
                    + F.cross_entropy(pt, tr["vid_action"][i]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
    m.train()
    for step in range(a.steps):
        with torch.autocast("cuda", torch.bfloat16):
            loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    m.eval()
    acc = lambda p, y: (p.argmax(1) == y).float().mean().item()  # noqa: E731
    with torch.no_grad():
        if a.what == "img":
            pa, pc = m(to_float(te["img"]))
            met = {"animal_acc": acc(pa, te["img_animal"]), "color_acc": acc(pc, te["img_color"])}
        else:
            pa, pc, pt = [torch.cat(o) for o in zip(*[m(to_float(te["vid"][k:k + 64])) for k in range(0, len(te["vid"]), 64)])]
            met = {"animal_acc": acc(pa, te["vid_animal"]), "color_acc": acc(pc, te["vid_color"]),
                   "action_acc": acc(pt, te["vid_action"])}
    rep = {"what": a.what, "steps": a.steps, "seconds": round(time.time() - t0, 1),
           "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
           "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), **met}
    torch.save({"model": m.state_dict()}, f"{a.out}/judge_{a.what}.pt")
    json.dump(rep, open(f"{a.out}/judge_{a.what}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()

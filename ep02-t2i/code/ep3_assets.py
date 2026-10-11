"""第 2 集(生图)的素材:全部来自同一份本集权重(第 2 集训练出的 final.pt),不重训。

  python ep3_assets.py --ckpt <W>/runs/ep02/final.pt --data <W>/data --judges <W>/judges --out <W>/runs/ep02_assets

产物都在 out/,数字、配置、权重 md5 汇总在 out/summary.json。提示「红色 小猫」,起点噪声种子 0(S.start_noise),30 步,CFG 3:
  1 steps/      一步步生成:第 0 步(纯噪声)到第 30 步每一步都解码存图(文件名是实际步号),另出每 3 步一张的横条;
                全来自同一次采样,末帧和 infer.py 同提示的单次推理逐像素一致(summary 里记两者 md5)
  2 lesson/     训练怎么出题:一条真实训练样本的目标图、噪声、抽到的时间 t、混合状态 z_t=(1-t)·噪声+t·目标、
                监督方向 v=目标-噪声(8x8 位置上的长度热力图)、模型在这道题上预测的方向、顺着预测一步跳到 t=1 的样子;
                另出 t=0..1 九档的混合横条
  3 sweep/      同权重、同噪声:采样步数 1/2/3/5/8/10/15/25/50、CFG 0/0.5/1/2/3/5/7/10 各一张,另各一张 6x8 网格(评测噪声种子 0);
                每档量画对率(240 张,6 动物x8 色x5),评测噪声种子 0/1/2 各一遍:第 i 题的起点噪声按种子 seed*10000+i 单独生成
  4 nearest/    最近邻:8 张生成图各配训练集最像的 3 张。距离口径:64x64x3 像素(0~255)展平后的欧氏距离,
                除以 sqrt(12288) 换成每个像素通道的均方根误差;基准:240 张独立测试图(种子不重叠)到训练集的最近邻距离
  5 vae/        压缩器往返:8 张测试图过预训练 FLUX.2 VAE 编码再解码与 PSNR——只用预训练 VAE,不经过麻雀模型
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import frozen
import sparrow as S
from parts import ANIMAL_ID, COLOR_ID, ID, ImageJudge, load_judge, to_float

dev = "cuda"
PROMPT, SEED, STEPS, CFG = ("红色", "小猫"), 0, 30, 3.0
EVAL_SEEDS = (0, 1, 2)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def big(u8, px=256):
    return Image.fromarray(np.asarray(u8.cpu() if torch.is_tensor(u8) else u8)).resize((px, px), Image.NEAREST)


def strip(imgs, labels, path, px=192, gap=4):
    out = Image.new("RGB", (len(imgs) * (px + gap) - gap, px + 20), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        out.paste(big(im, px), (k * (px + gap), 0))
        d.text((k * (px + gap) + 4, px + 4), str(lab), fill=(0, 0, 0))
    out.save(path)


def heat(v):
    """[64, 32] 的方向 -> 8x8 每个位置上方向的长度,灰度图(越亮越长)。"""
    m = v.float().norm(dim=-1).view(8, 8)
    return (m / m.max() * 255).byte()[..., None].repeat(1, 1, 3)


def main():
    ap = argparse.ArgumentParser()
    for k in ("ckpt", "data", "judges", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    for sub in ("steps", "lesson", "sweep", "nearest", "vae"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    model, lays, stat = S.load(a.ckpt)
    lay = lays["t2i"]
    vae = frozen.ImageVAE()
    judge = load_judge(ImageJudge, f"{a.judges}/judge_img.pt", dev)
    image = lambda x: torch.cat([vae.decode(S.denorm(stat, "img", x[s:e])) for s, e in S.chunks(len(x), 128)])  # noqa: E731
    prompt = lambda ai, ci: {"C": COLOR_ID.to(dev)[torch.as_tensor(ci, device=dev)],  # noqa: E731
                             "A": ANIMAL_ID.to(dev)[torch.as_tensor(ai, device=dev)]}
    one = {"C": torch.tensor([ID[PROMPT[0]]], device=dev), "A": torch.tensor([ID[PROMPT[1]]], device=dev)}
    x0 = S.start_noise(lay, 1, SEED)
    rep = {"ckpt": a.ckpt, "ckpt_md5": md5(a.ckpt), "prompt": " ".join(PROMPT), "noise_seed": SEED, "steps": STEPS, "cfg": CFG}

    # ---- 1 一步步生成
    trace = []
    S.generate(model, lays, "t2i", one, 1, cfg=CFG, steps=STEPS, x0=x0, trace=trace)
    frames = [image(x)[0] for x in trace]                              # 一张一张解码,和 infer.py 的单张解码同一批大小
    for k, fr in enumerate(frames):
        big(fr).save(f"{a.out}/steps/step{k:02d}.png")
    strip(frames[::3], [f"step {k}" for k in range(0, STEPS + 1, 3)], f"{a.out}/steps/strip_every3.png")
    check = f"{a.out}/steps/infer_same_prompt.png"
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py"),
                    a.ckpt, "t2i", " ".join(PROMPT), check], check=True)
    rep["last_step_md5"], rep["infer_md5"] = md5(f"{a.out}/steps/step{STEPS:02d}.png"), md5(check)
    rep["last_step_equals_infer"] = rep["last_step_md5"] == rep["infer_md5"]

    # ---- 2 训练怎么出题:第 0 张训练图,噪声种子 1,时间按训练时的 logit-normal 抽一个
    raw = np.load(f"{a.data}/world_train.npz")
    lab = {"animal": int(raw["img_animal"][0]), "color": int(raw["img_color"][0])}
    u8 = torch.from_numpy(raw["img"][:1]).to(dev)
    with torch.autocast("cuda", torch.bfloat16):                        # 和 encode.py 编缓存时同一精度
        x1 = S.norm(stat, "img", vae.encode(u8).float())
    g = torch.Generator(device=dev).manual_seed(1)
    eps = torch.randn(x1.shape, device=dev, generator=g)
    t = torch.sigmoid(torch.randn(1, device=dev, generator=g))
    zt = (1 - t) * eps + t * x1
    v = x1 - eps
    tok = S.fill_tok(lay, prompt([lab["animal"]], [lab["color"]]), 1)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        vp = model(lay, tok, [zt], t)[1].float()
    jump = zt + (1 - t) * vp
    pics = image(torch.cat([x1, eps, zt, jump]))
    for name, im in zip(("target", "noise", "mixed_zt", "jump_along_prediction"), pics):
        big(im).save(f"{a.out}/lesson/{name}.png")
    big(heat(v[0])).save(f"{a.out}/lesson/direction_true.png")
    big(heat(vp[0])).save(f"{a.out}/lesson/direction_predicted.png")
    ts = torch.linspace(0, 1, 9, device=dev)
    strip(image(torch.cat([(1 - s) * eps + s * x1 for s in ts])), [f"t={s:.3f}" for s in ts.tolist()], f"{a.out}/lesson/mix_t_0_to_1.png")
    rep["lesson"] = {"train_index": 0, "label": lab, "noise_seed": 1, "t": round(t.item(), 4),
                     "loss_mse": round(torch.mean((vp - v) ** 2).item(), 4),
                     "cos_pred_vs_true": round(torch.nn.functional.cosine_similarity(vp.flatten(), v.flatten(), 0).item(), 4)}

    # ---- 3 扫步数与 CFG
    ga = torch.arange(6, device=dev).repeat_interleave(40)
    gc = torch.arange(8, device=dev).repeat_interleave(5).repeat(6)
    gx0 = {sd: torch.cat([S.start_noise(lay, 1, sd * 10000 + i) for i in range(len(ga))]) for sd in EVAL_SEEDS}

    def score(cfg, steps, sd):
        xs = [S.generate(model, lays, "t2i", prompt(ga[s:e], gc[s:e]), e - s, cfg=cfg, steps=steps, x0=gx0[sd][s:e]) for s, e in S.chunks(len(ga), 120)]
        imgs = image(torch.cat(xs))
        pa, pc = judge(to_float(imgs))
        ok_a, ok_c = pa.argmax(1) == ga, pc.argmax(1) == gc
        return {"animal_acc": ok_a.float().mean().item(), "color_acc": ok_c.float().mean().item(),
                "both_acc": (ok_a & ok_c).float().mean().item()}, imgs
    rep["sweep"] = {"acc_grid": "240 张:6 动物 x 8 颜色 x 5;评测噪声种子 0/1/2,第 i 题起点噪声种子 = seed*10000+i", "steps": {}, "cfg": {}}
    for key, vals in (("steps", (1, 2, 3, 5, 8, 10, 15, 25, 50)), ("cfg", (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0))):
        ims = []
        for val in vals:
            cfg, steps = (CFG, val) if key == "steps" else (val, STEPS)
            ims.append(image(S.generate(model, lays, "t2i", one, 1, cfg=cfg, steps=steps, x0=x0))[0])
            big(ims[-1]).save(f"{a.out}/sweep/{key}_{val:g}.png")
            by = {}
            for sd in EVAL_SEEDS:
                by[sd], gi = score(cfg, steps, sd)
                if sd == 0:                                                 # 每个组合取第 1 张拼 6x8 网格(行=动物,列=颜色)
                    grid = gi.view(6, 8, 5, 64, 64, 3)[:, :, 0].permute(0, 2, 1, 3, 4).reshape(6 * 64, 8 * 64, 3)
                    Image.fromarray(grid.cpu().numpy()).resize((8 * 128, 6 * 128), Image.NEAREST).save(f"{a.out}/sweep/{key}_{val:g}_grid.png")
            both = [by[sd]["both_acc"] for sd in EVAL_SEEDS]
            rep["sweep"][key][f"{val:g}"] = {"cfg": cfg, "steps": steps, "by_seed": {str(sd): by[sd] for sd in EVAL_SEEDS},
                                             "both_mean": sum(both) / len(both), "both_min": min(both), "both_max": max(both)}
        strip(ims, [f"{key} {val:g}" for val in vals], f"{a.out}/sweep/{key}_strip.png")

    # ---- 4 最近邻
    train = torch.from_numpy(raw["img"]).flatten(1)                   # 留在内存,分块搬上卡
    test = torch.from_numpy(np.load(f"{a.data}/world_test.npz")["img"][:240]).to(dev).flatten(1).float()
    gen = score(CFG, STEPS, EVAL_SEEDS[0])[1].flatten(1).float()

    def nearest(q, k):
        best_d = torch.full((len(q), k), float("inf"), device=dev)
        best_i = torch.zeros((len(q), k), dtype=torch.long, device=dev)
        for s, e in S.chunks(len(train), 8192):
            d = torch.cat([best_d, torch.cdist(q, train[s:e].to(dev).float())], 1)
            i = torch.cat([best_i, torch.arange(s, e, device=dev).expand(len(q), -1)], 1)
            best_d, j = d.topk(k, largest=False)
            best_i = i.gather(1, j)
        return best_d / (q.shape[1] ** 0.5), best_i
    dg, ig = nearest(gen, 3)
    dt, _ = nearest(test, 1)
    pick = list(range(0, 240, 30))
    rows = Image.new("RGB", (4 * 196, 8 * 196), (255, 255, 255))
    for r, j in enumerate(pick):
        for c, im in enumerate([gen[j]] + [train[ig[j, m]] for m in range(3)]):
            rows.paste(big(im.view(64, 64, 3).byte(), 192), (c * 196, r * 196))
    rows.save(f"{a.out}/nearest/gen_vs_top3_train.png")
    q = lambda d: {p: round(float(np.percentile(d.cpu().numpy(), p)), 1) for p in (0, 5, 50, 95)}  # noqa: E731
    rep["nearest"] = {"metric": "像素 L2 / sqrt(12288) = 每个像素通道的均方根误差(0~255)", "n": 240,
                      "generated": {"mean": round(dg[:, 0].mean().item(), 1), "pct": q(dg[:, 0])},
                      "test_baseline": {"mean": round(dt[:, 0].mean().item(), 1), "pct": q(dt[:, 0])}}
    with open(f"{a.out}/nearest/distances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "rmse_to_nearest_train"])
        w.writerows([["generated", round(x, 2)] for x in dg[:, 0].tolist()] + [["test", round(x, 2)] for x in dt[:, 0].tolist()])

    # ---- 5 压缩器往返(只用预训练 VAE)
    raw8 = torch.from_numpy(np.load(f"{a.data}/world_test.npz")["img"][:8]).to(dev)
    rec = vae.decode(vae.encode(raw8))
    mse = ((raw8.float() - rec.float()) ** 2).flatten(1).mean(1)
    rep["vae_roundtrip"] = {"note": "只用预训练 FLUX.2 VAE,不经过麻雀模型",
                            "psnr_dB": [round(x, 1) for x in (10 * torch.log10(255 ** 2 / mse)).tolist()]}
    strip([im for pair in zip(raw8, rec) for im in pair], ["orig", "vae"] * 8, f"{a.out}/vae/roundtrip_8pairs.png", px=128)

    json.dump(rep, open(f"{a.out}/summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if not rep["last_step_equals_infer"]:
        raise SystemExit("一步步生成的末帧和 infer.py 单次推理不一致")


if __name__ == "__main__":
    main()

"""对照实验「两种表示方案」:同一主干、同样超参,只换生成用的表示,只训这一样本事(不是主线权重)。
第 3 集比图(潜变量 vs 像素),第 4 集比声音(潜变量 vs 波形);每次只训一个版本,两版各占一张卡同时跑,耗时与显存才可比。

  python ep1_assets.py --data D --cache C --judges J --out O --mod img|aud --space latent|raw [--steps N] [--seed S] [--rng paired] [--eval_seed E]

  img latent  FLUX.2 潜变量 8x8 = 64 个 32 维向量            img raw  像素 64x64x3 切 8x8 块 = 64 个 192 维向量
  aud latent  Mimi 潜变量 26 帧、每帧拆 4 个 = 104 个 128 维   aud raw  波形 2.08 秒 49920 个数切 104 块 = 104 个 480 维向量
训练数据整份放在卡上(数据占显存单列);各维按训练集的均值/标准差归一化(像素版照原样缩到 [-1, 1])。
图:240 张(6 动物 x 8 颜色 x 5)交图判卷器判动物+颜色;声音:240 条(6 x 40)解码成波形交 Paraformer 听写。起点噪声按题目序号生成(同 sparrow.py)。
O/summary.json:训练耗时与每步秒数、训练显存峰值、数据占显存、进出主干的两个投影的参数量、完整序列长度、正确率、随机协议与公共主干初始哈希;
O/init.json 起跑就写(公共主干初始哈希,成对的两版用它核对);O/loss.csv 每 100 步的训练损失;O/samples/ 同题样例。
随机协议同 sparrow.py 的 --rng:paired 时两版同种子的公共主干初始化相同、抽到的样本序号与时间 t 相同,只有表示不同。
"""
import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from PIL import Image

import frozen
import sparrow as S
from encode import train_wavs
from parts import ANIMAL_ID, ANIMAL_ZH, ANIMALS, COLOR_ID, SR, ImageJudge, load_judge, to_float, to_u8

dev = S.dev
S.MODS["pix"] = (192, 64, 1, True)                       # 像素版:8x8 一块,64 块,每块 8*8*3=192 个数
S.MODS["wav"] = (480, 104, 1, True)                      # 波形版:49920 个数切 104 块,每块 480 个(和 Mimi 版同样 104 个向量)
S.TEMPLATES["t2p"] = ["<bos>", "C", "A", "<img>", ("pix", True)]
S.TEMPLATES["t2w"] = ["<bos>", "A", "<aud>", ("wav", True)]
S.GEN |= {"t2p", "t2w"}
TASK = {("img", "latent"): "t2i", ("img", "raw"): "t2p", ("aud", "latent"): "t2a", ("aud", "raw"): "t2w"}
PAIR = {"t2p": "t2i", "t2w": "t2a"}                       # 原始表示版与潜变量版共用一个配对身份:随机流按它取,两版抽到同样的样本、时间、去条件


def pix_tokens(u8):                                      # [B, 64, 64, 3] uint8 -> [B, 64, 192] in [-1, 1]
    B = u8.shape[0]
    return to_float(u8).view(B, 3, 8, 8, 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(B, 64, 192)


def pix_image(x):
    B = x.shape[0]
    return to_u8(x.float().view(B, 8, 8, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 64, 64))


class OneData:
    """只装这一样本事要的东西,整份放卡上:标签 + 潜变量 / 像素 / 波形(后两者是原始数据,不经压缩器)。"""

    def __init__(self, a):
        self.mod, self.space, self.task = a.mod, a.space, TASK[(a.mod, a.space)]
        if a.mod == "img":
            lab = np.load(f"{a.cache}/labels.npz")
            self.animal = torch.from_numpy(lab["train.img_animal"]).to(dev)
            self.color = torch.from_numpy(lab["train.img_color"]).to(dev)
            if a.space == "raw":
                self.x = torch.from_numpy(np.load(f"{a.data}/world_train.npz")["img"]).to(dev)
                return
            x = S.tokens("img_lat", torch.from_numpy(np.load(f"{a.cache}/img.npz")["train.img_lat"]).float())
        else:
            voice = np.load(f"{a.data}/voice.npz")
            self.animal = torch.from_numpy(np.concatenate([voice["animal"][~voice["test"]]] * 4)).to(dev)  # 和 encode.py 一样每条 4 份
            if a.space == "raw":
                x = torch.from_numpy(train_wavs(voice)[:len(self.animal)]).float().view(len(self.animal), 104, 480)
            else:
                lab = np.load(f"{a.cache}/labels.npz")
                lat = torch.from_numpy(np.load(f"{a.cache}/aud.npz")["voice.aud_lat"][~lab["voice.test"]])
                x = S.tokens("aud_lat", lat.float())
        flat = x.flatten(0, 1)
        self.mu, self.sd = flat.mean(0).to(dev), flat.std(0).to(dev) + 1e-4
        self.x = ((x.to(dev) - self.mu) / self.sd).half()

    def get(self, task, B, gen=None):
        i = torch.randint(0, len(self.animal), (B,), device=dev, generator=gen)
        slots = {"A": ANIMAL_ID.to(dev)[self.animal[i]]}
        if self.mod == "img":
            slots["C"] = COLOR_ID.to(dev)[self.color[i]]
        return slots, [pix_tokens(self.x[i]) if self.task == "t2p" else self.x[i]]

    def decode(self, x, codec):
        """生成的向量 -> 图 [B, 64, 64, 3](uint8)或波形 [B, 49920]。"""
        if self.task == "t2p":
            return pix_image(x)
        x = x.float() * self.sd + self.mu
        if self.task == "t2w":
            return x.flatten(1).clamp(-1, 1)
        name = "img_lat" if self.mod == "img" else "aud_lat"
        return torch.cat([codec.decode(S.untokens(name, x[s:e])) for s, e in S.chunks(len(x), 64)])


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "cache", "judges", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--mod", required=True, choices=["img", "aud"])
    ap.add_argument("--space", required=True, choices=["latent", "raw"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0, help="训练种子")
    ap.add_argument("--rng", default="legacy", choices=["legacy", "paired"], help="随机协议,同 sparrow.py")
    ap.add_argument("--eval_seed", type=int, default=0, help="评测噪声种子:第 i 题起点噪声按 eval_seed*10000+i 生成")
    a = ap.parse_args()
    S.fresh_out(a.out)
    os.makedirs(f"{a.out}/samples", exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    S.split_audio(4)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data = OneData(a)
    lays = {t: S.Layout(t) for t in ("t2i", data.task)}       # 生成时无条件那一路:图用同模板,声音用自己的模板(没有颜色词)
    model = S.Sparrow(seed=a.seed if a.rng == "paired" else None).to(dev)
    built = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init = S.identity(a, model, built, "random_init")
    S.write_json(init, f"{a.out}/init.json")
    data_mb = round(data.x.nbytes / 1e6)
    args = SimpleNamespace(tasks=[data.task], steps=a.steps, lr=6e-4, bs=512, bs_vid=128, bs_task={}, ema=0.999, weights={},
                           eval_points=[], seed=a.seed, rng=a.rng, stream_alias=PAIR)
    sec, log, train_peak, _ = S.train(args, model, data, lays)
    mod = S.TEMPLATES[data.task][-1][0]
    rep = {"mod": a.mod, "space": a.space, "task": data.task, "stream_identity": PAIR.get(data.task, data.task), "seed": a.seed,
           "rng": a.rng, "init": init, "train_rows": len(data.animal), "eval_seed": a.eval_seed,
           "steps": a.steps, "train_seconds": sec, "sec_per_step": round(sec / a.steps, 4), "train_peak_mem_GB": train_peak,
           "data_on_gpu_MB": data_mb,
           "vectors": list(S.MODS[mod][1:3]), "vector_dim": S.MODS[mod][0], "seq_len": lays[data.task].L,
           "proj_in_params": sum(p.numel() for p in model.inp[mod].parameters()),
           "proj_out_params": sum(p.numel() for p in model.flow_head[mod].parameters()),
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    with open(f"{a.out}/loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows([[r["step"], r[data.task]] for r in log])
    model.eval()
    lay = lays[data.task]
    if a.mod == "img":                                      # 240 张:6 动物 x 8 颜色 x 5
        ga = torch.arange(6, device=dev).repeat_interleave(40)
        gc = torch.arange(8, device=dev).repeat_interleave(5).repeat(6)
        slots = {"C": COLOR_ID.to(dev)[gc], "A": ANIMAL_ID.to(dev)[ga]}
        codec = frozen.ImageVAE() if a.space == "latent" else None
    else:                                                   # 240 条:6 动物 x 40
        ga = torch.arange(6, device=dev).repeat_interleave(40)
        slots = {"A": ANIMAL_ID.to(dev)[ga]}
        codec = frozen.Voice() if a.space == "latent" else None
    x0 = S.eval_noise(lay, len(ga), a.eval_seed)
    xs = [S.generate(model, lays, data.task, {k: v[s:e] for k, v in slots.items()}, e - s, x0=x0[s:e]) for s, e in S.chunks(len(ga), 120)]
    out = data.decode(torch.cat(xs), codec)
    if a.mod == "img":
        judge = load_judge(ImageJudge, f"{a.judges}/judge_img.pt", dev)
        pa, pc = [o.argmax(1) for o in judge(to_float(out))]
        rep.update({"n": len(ga), "animal_acc": (pa == ga).float().mean().item(), "color_acc": (pc == gc).float().mean().item(),
                    "both_acc": ((pa == ga) & (pc == gc)).float().mean().item()})
        strip = Image.new("RGB", (6 * 196 - 4, 192), (255, 255, 255))
        for k in range(6):                                  # 每种动物取第 1 张(红色)
            strip.paste(Image.fromarray(out[k * 40].cpu().numpy()).resize((192, 192), Image.NEAREST), (k * 196, 0))
        strip.save(f"{a.out}/samples/gen_{a.space}_6animals.png")
        for k in range(6):                                  # 六张也各存一张(第 0 张是红色小猫)
            Image.fromarray(out[k * 40].cpu().numpy()).resize((256, 256), Image.NEAREST).save(
                f"{a.out}/samples/gen_{a.space}_{k}_红色{ANIMAL_ZH[ANIMALS[k]]}.png")
    else:
        from asr import Paraformer
        judge = Paraformer(dev)
        same, exact, heard = judge.hits(out, SR, [ANIMAL_ZH[ANIMALS[int(k)]] for k in ga])
        rep.update({"n": len(ga), "judge": judge.name, "sound_acc": float(same.mean()), "exact_acc": float(exact.mean()),
                    "sound_by_word": {ANIMAL_ZH[ANIMALS[k]]: float(same[k * 40:(k + 1) * 40].mean()) for k in range(6)},
                    "misheard": [h for h, k in zip(heard, same) if not k][:20]})
        for k in range(6):
            sf.write(f"{a.out}/samples/gen_{a.space}_{k}_{ANIMAL_ZH[ANIMALS[k]]}.wav", out[k * 40].cpu().numpy(), SR)
    S.write_json(rep, f"{a.out}/summary.json")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

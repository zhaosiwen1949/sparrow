"""卡通小世界:6 种小动物 × 8 色 × 4 背景的程序化数据生成器。

纯 Python + numpy + PIL,给定 rng 完全可复现。

    sample(rng, animal=None, hd=False) -> dict(image=64x64x3 uint8, animal, color, bg, facing, scale, x, y[, hd])
    animate(rng, action, animal=None, hd=False) -> dict(frames=[16 个 64x64x3], ...[, hd_frames])
        action ∈ run_left / run_right / jump / turn

画法:在 512×512 上用多边形画(形状坐标写在「单位空间」,朝右、脚底 y=80),
缩到 256×256 得高清图,缩到 64×64 得训练图(超采样抗锯齿)。
"""
import math

import numpy as np
from PIL import Image, ImageDraw

ANIMALS = ["cat", "dog", "rabbit", "bird", "fish", "bear"]
ANIMAL_ZH = {"cat": "小猫", "dog": "小狗", "rabbit": "小兔", "bird": "小鸟", "fish": "小鱼", "bear": "小熊"}
COLORS = {
    "red": (232, 84, 84), "orange": (246, 152, 60), "yellow": (246, 212, 72), "green": (112, 192, 92),
    "blue": (84, 152, 232), "purple": (164, 112, 212), "pink": (246, 152, 192), "brown": (164, 112, 72),
}
BGS = ["sky", "grass", "water", "plain"]
HABITAT = {"fish": ["water"], "bird": ["sky", "grass", "plain"],
           "cat": ["grass", "plain"], "dog": ["grass", "plain"], "rabbit": ["grass", "plain"], "bear": ["grass", "plain"]}
ACTIONS = ["run_left", "run_right", "jump", "turn"]
PLAIN = [(250, 240, 220), (220, 242, 228), (232, 226, 248), (252, 226, 214), (236, 236, 236)]
S = 512          # 作画分辨率
FRAMES = 16


def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def ellipse_pts(cx, cy, rx, ry, ang=0.0, n=48):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cx + x * c - y * s, cy + x * s + y * c))
    return out


class Pen:
    """单位空间 → 画布像素;每个形状自带深色描边。"""

    def __init__(self, draw, k, ink):
        self.d, self.k, self.ink = draw, k, ink
        self.w = max(1, int(round(6 * k)))

    def P(self, pts):
        return [(S / 2 + x * self.k, S / 2 + y * self.k) for x, y in pts]

    def shape(self, pts, fill, outline=True):
        p = self.P(pts)
        self.d.polygon(p, fill=fill)
        if outline:
            self.d.line(p + [p[0], p[1]], fill=self.ink, width=self.w, joint="curve")

    def ell(self, cx, cy, rx, ry, fill, ang=0.0, outline=True):
        self.shape(ellipse_pts(cx, cy, rx, ry, ang), fill, outline)

    def line(self, pts, fill, w):
        p = self.P(pts)
        ww = max(1, int(round(w * self.k)))
        self.d.line(p, fill=fill, width=ww, joint="curve")
        r = ww / 2
        for x, y in (p[0], p[-1]):
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def stroke(self, pts, fill, w):
        """带描边的粗线(尾巴)。"""
        self.line(pts, self.ink, w + 12)
        self.line(pts, fill, w)

    def eye(self, cx, cy, r):
        self.ell(cx, cy, r * 0.82, r, (34, 30, 40), outline=False)
        self.ell(cx - r * 0.28, cy - r * 0.38, r * 0.36, r * 0.36, (255, 255, 255), outline=False)
        self.ell(cx + r * 0.3, cy + r * 0.4, r * 0.16, r * 0.16, (255, 255, 255), outline=False)


def palette(color):
    body = COLORS[color]
    return dict(body=body, ink=mix(body, (30, 20, 40), 0.72), dark=mix(body, (40, 30, 50), 0.22),
                light=mix(body, (255, 255, 255), 0.6), cheek=mix(body, (255, 80, 120), 0.5),
                pink=(250, 160, 180))


def ear_tri(pen, hx, hy, r, ang, length, half, fill, inner):
    def pt(a, rr):
        return (hx + rr * math.cos(math.radians(a)), hy + rr * math.sin(math.radians(a)))
    pen.shape([pt(ang - half, r * 0.85), pt(ang, r + length), pt(ang + half, r * 0.85)], fill)
    pen.shape([pt(ang - half * 0.5, r * 0.9), pt(ang, r + length * 0.7), pt(ang + half * 0.5, r * 0.9)], inner, outline=False)


def legs(pen, pal, pose, hips, y, rx, ry, far):
    amp = 11 * pose["run"]
    for i, hx in enumerate(hips):
        sw = math.sin(pose["phase"] + (0 if (i == 0) != far else math.pi)) * amp + pose["tuck"] * (8 if i else -8)
        pen.ell(hx + sw, y - abs(sw) * 0.25 - pose["tuck"] * 6, rx, ry, pal["dark"] if far else pal["body"])


def face(pen, pal, fx, fy, r, gap):
    pen.eye(fx - gap, fy, r)
    pen.eye(fx + gap, fy, r)
    pen.ell(fx - gap - 8, fy + r + 9, 9, 6, pal["cheek"], outline=False)
    pen.ell(fx + gap + 8, fy + r + 9, 9, 6, pal["cheek"], outline=False)


def smile(pen, cx, cy, w=7):
    pen.line([(cx - w, cy), (cx - w / 2, cy + 4), (cx, cy)], pen.ink, 3)
    pen.line([(cx, cy), (cx + w / 2, cy + 4), (cx + w, cy)], pen.ink, 3)


def draw_cat(pen, pal, pose):
    wag = math.sin(pose["phase"]) * 6
    legs(pen, pal, pose, [-45, 12], 62, 11, 18, True)
    pen.stroke([(-60, 30), (-82, 10), (-88 + wag, -20), (-78 + wag, -40)], pal["body"], 12)
    pen.ell(-15, 32, 55, 36, pal["body"])
    pen.ell(-5, 42, 30, 18, pal["light"], outline=False)
    legs(pen, pal, pose, [-35, 22], 64, 11, 17, False)
    hx, hy = 35, -18
    ear_tri(pen, hx, hy, 44, -125, 30, 22, pal["body"], pal["pink"])
    ear_tri(pen, hx, hy, 44, -55, 30, 22, pal["body"], pal["pink"])
    pen.ell(hx, hy, 46, 42, pal["body"])
    face(pen, pal, hx + 8, hy - 4, 11, 17)
    pen.shape([(hx + 3, hy + 10), (hx + 13, hy + 10), (hx + 8, hy + 16)], pal["pink"])
    smile(pen, hx + 8, hy + 17)
    for dy in (-4, 4):
        pen.line([(hx + 34, hy + 12 + dy), (hx + 60, hy + 8 + dy * 2.2)], pen.ink, 2.5)
        pen.line([(hx - 18, hy + 12 + dy), (hx - 44, hy + 8 + dy * 2.2)], pen.ink, 2.5)


def draw_dog(pen, pal, pose):
    wag = math.sin(pose["phase"] * 2) * 10
    legs(pen, pal, pose, [-45, 12], 62, 12, 18, True)
    pen.stroke([(-62, 22), (-80 + wag * 0.3, 0), (-84 + wag, -18)], pal["body"], 12)
    pen.ell(-15, 32, 56, 36, pal["body"])
    pen.ell(-5, 42, 30, 18, pal["light"], outline=False)
    legs(pen, pal, pose, [-35, 22], 64, 12, 17, False)
    hx, hy = 32, -18
    pen.ell(hx + 22, hy - 30, 13, 26, pal["dark"], ang=-35)          # 远侧耳
    pen.ell(hx, hy, 44, 40, pal["body"])
    pen.ell(hx + 30, hy + 16, 24, 17, pal["light"])                    # 口鼻
    pen.ell(hx + 50, hy + 8, 10, 8, (40, 32, 40), outline=False)       # 鼻头
    pen.ell(hx + 47, hy + 5, 3, 2, (255, 255, 255), outline=False)
    pen.ell(hx + 32, hy + 32, 7, 8, (240, 110, 130))                  # 舌头
    pen.eye(hx - 6, hy - 10, 10)
    pen.eye(hx + 20, hy - 10, 10)
    pen.ell(hx - 14, hy + 14, 9, 6, pal["cheek"], outline=False)
    pen.ell(hx - 34, hy + 6, 15, 30, pal["dark"], ang=18)             # 垂耳


def draw_rabbit(pen, pal, pose):
    legs(pen, pal, pose, [-42, 12], 62, 12, 17, True)
    pen.ell(-66, 22, 15, 15, (255, 255, 255))                          # 团子尾
    pen.ell(-15, 32, 52, 36, pal["body"])
    pen.ell(-5, 42, 28, 18, pal["light"], outline=False)
    legs(pen, pal, pose, [-34, 22], 64, 13, 16, False)
    hx, hy = 32, -8
    for ex, ang in ((hx - 14, -12), (hx + 16, 10)):
        pen.ell(ex, hy - 64, 13, 36, pal["body"], ang=ang)
        pen.ell(ex, hy - 60, 6, 26, pal["pink"], ang=ang, outline=False)
    pen.ell(hx, hy, 42, 38, pal["body"])
    face(pen, pal, hx + 6, hy - 4, 10, 16)
    pen.ell(hx + 6, hy + 11, 5, 4, pal["pink"])
    pen.shape([(hx + 2, hy + 21), (hx + 10, hy + 21), (hx + 10, hy + 28), (hx + 2, hy + 28)], (255, 255, 255))
    smile(pen, hx + 6, hy + 16, 6)


def draw_bear(pen, pal, pose):
    legs(pen, pal, pose, [-45, 14], 60, 15, 20, True)
    pen.ell(-72, 18, 10, 10, pal["body"])
    pen.ell(-15, 30, 60, 40, pal["body"])
    pen.ell(-5, 40, 34, 22, pal["light"], outline=False)
    legs(pen, pal, pose, [-35, 24], 62, 15, 19, False)
    hx, hy = 32, -18
    for a in (-135, -48):
        ex, ey = hx + 42 * math.cos(math.radians(a)), hy + 40 * math.sin(math.radians(a))
        pen.ell(ex, ey, 16, 16, pal["body"])
        pen.ell(ex, ey, 8, 8, pal["light"], outline=False)
    pen.ell(hx, hy, 46, 42, pal["body"])
    pen.ell(hx + 14, hy + 16, 22, 15, pal["light"])
    pen.ell(hx + 16, hy + 9, 9, 6, (40, 32, 40), outline=False)
    pen.ell(hx + 14, hy + 7, 3, 2, (255, 255, 255), outline=False)
    smile(pen, hx + 16, hy + 17, 6)
    pen.eye(hx - 12, hy - 10, 9)
    pen.eye(hx + 30, hy - 10, 9)
    pen.ell(hx - 24, hy + 10, 9, 6, pal["cheek"], outline=False)
    pen.ell(hx + 40, hy + 8, 7, 5, pal["cheek"], outline=False)


def draw_bird(pen, pal, pose):
    orange = (250, 160, 40)
    flap = math.sin(pose["phase"] * 2) * (25 if pose["run"] or pose["tuck"] else 5)
    for fx in (-12, 12):
        pen.line([(fx, 50), (fx, 76)], orange, 5)
        pen.line([(fx - 7, 78), (fx + 9, 78)], orange, 5)
    pen.shape([(-45, 10), (-84, -8), (-78, 12), (-84, 30)], pal["dark"])  # 尾羽
    pen.ell(0, 8, 56, 50, pal["body"])
    pen.ell(8, 26, 36, 26, pal["light"], outline=False)
    pen.ell(-8, -44, 7, 12, pal["body"], ang=-20)                      # 呆毛
    pen.ell(4, -46, 6, 10, pal["body"], ang=15)
    pen.shape([(44, -12), (76, -2), (44, 10)], orange)                  # 喙
    pen.eye(8, -12, 10)
    pen.eye(32, -12, 10)
    pen.ell(-2, 8, 9, 6, pal["cheek"], outline=False)
    pen.ell(-18, 22, 30, 18, pal["dark"], ang=-25 + flap)               # 翅膀


def draw_fish(pen, pal, pose):
    w = math.sin(pose["phase"]) * (14 if pose["run"] else 5)
    pen.shape([(-45, 0), (-92, -34 + w), (-80, w), (-92, 34 + w)], pal["dark"])   # 尾鳍
    pen.shape([(-25, -30), (-5, -62), (25, -36)], pal["dark"])                      # 背鳍
    pen.shape([(0, 34), (-18, 52), (18, 38)], pal["dark"])                           # 腹鳍
    pen.ell(0, 0, 62, 40, pal["body"])
    pen.ell(-18, 0, 7, 32, pal["light"], outline=False)
    pen.ell(4, 0, 6, 34, pal["light"], outline=False)
    pen.ell(8, 18, 36, 14, pal["light"], outline=False)
    pen.ell(-8, 12, 12, 7, pal["dark"], ang=30 + w, outline=False)      # 胸鳍
    pen.eye(34, -10, 13)
    pen.ell(40, 14, 8, 5, pal["cheek"], outline=False)
    pen.line([(54, 10), (60, 14), (56, 18)], pen.ink, 3)


DRAW = dict(cat=draw_cat, dog=draw_dog, rabbit=draw_rabbit, bear=draw_bear, bird=draw_bird, fish=draw_fish)


# ---------------------------------------------------------------- 背景

def _grad(top, bot, h=S):
    t = np.linspace(0, 1, h)[:, None, None]
    col = np.array(top)[None, None] * (1 - t) + np.array(bot)[None, None] * t
    return np.repeat(col, S, axis=1)


def make_bg(rng, bg):
    """返回 (512 画布, 地面 y(256 坐标)或 None)。"""
    ground = None
    if bg == "plain":
        img = Image.new("RGB", (S, S), PLAIN[rng.integers(len(PLAIN))])
        ground = float(rng.uniform(200, 230))
        return img, ground
    if bg == "water":
        arr = _grad((150, 215, 245), (40, 110, 190))
    else:
        arr = _grad((140, 200, 250), (215, 238, 255))
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    if bg == "sky":
        for _ in range(int(rng.integers(2, 4))):
            cx, cy, r = rng.uniform(0, S), rng.uniform(40, S - 40), rng.uniform(30, 55)
            for dx, dy, rr in ((0, 0, 1), (-0.9, 0.2, 0.7), (0.9, 0.25, 0.65)):
                x, y, R = cx + dx * r, cy + dy * r, r * rr
                d.ellipse([x - R, y - R, x + R, y + R], fill=(255, 255, 255))
    elif bg == "grass":
        ground = float(rng.uniform(190, 215))
        gy = int(ground * 2 - 20)
        d.rectangle([0, gy, S, S], fill=(120, 200, 100))
        d.rectangle([0, gy, S, gy + 8], fill=(100, 180, 85))
        for _ in range(14):
            x, y = rng.uniform(0, S), rng.uniform(gy + 20, S)
            d.line([(x, y), (x - 6, y - 16)], fill=(80, 160, 70), width=4)
            d.line([(x, y), (x + 6, y - 16)], fill=(80, 160, 70), width=4)
    else:
        for _ in range(4):
            y = rng.uniform(30, S - 60)
            ph = rng.uniform(0, 6.28)
            pts = [(x, y + 6 * math.sin(x / 30 + ph)) for x in range(0, S + 8, 8)]
            d.line(pts, fill=(190, 230, 250), width=3)
        d.rectangle([0, S - 40, S, S], fill=(236, 214, 160))
        for _ in range(5):
            x, y, r = rng.uniform(0, S), rng.uniform(20, S - 60), rng.uniform(5, 11)
            d.ellipse([x - r, y - r, x + r, y + r], outline=(225, 245, 255), width=3)
    return img, ground


# ---------------------------------------------------------------- 合成

def render(scene, pose, cx, cy, facing, sx=1.0, shadow_y=None):
    """scene: bg 画布等;cx, cy 是 256 坐标下身体中心。返回 512 画布。"""
    pal = palette(scene["color"])
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pen = Pen(ImageDraw.Draw(layer), 2 * scene["scale"], pal["ink"])
    DRAW[scene["animal"]](pen, pal, pose)
    if facing == "left":
        layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
    if sx < 0.999:
        nw = max(2, int(S * sx))
        squeezed = layer.resize((nw, S), Image.BILINEAR)
        layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        layer.paste(squeezed, ((S - nw) // 2, 0))
    img = scene["canvas"].copy()
    if shadow_y is not None:
        sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        rx, ry = 60 * scene["scale"] * 2 * max(sx, 0.4), 9 * scene["scale"] * 2
        x, y = cx * 2, shadow_y * 2
        ImageDraw.Draw(sh).ellipse([x - rx, y - ry, x + rx, y + ry], fill=(0, 0, 0, 50))
        img = Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB")
    img.paste(layer, (int(round(cx * 2 - S / 2)), int(round(cy * 2 - S / 2))), layer)
    return img


def _scene(rng, animal, scale_range):
    animal = animal or ANIMALS[rng.integers(len(ANIMALS))]
    color = list(COLORS)[rng.integers(len(COLORS))]
    bgs = HABITAT[animal]
    bg = bgs[rng.integers(len(bgs))]
    canvas, ground = make_bg(rng, bg)
    scale = float(rng.uniform(*scale_range))
    land = animal not in ("fish",) and ground is not None
    return dict(animal=animal, color=color, bg=bg, scale=scale, canvas=canvas, ground=ground if land else None)


def _outputs(img512, hd):
    small = np.asarray(img512.resize((64, 64), Image.LANCZOS))
    big = np.asarray(img512.resize((256, 256), Image.LANCZOS)) if hd else None
    return small, big


REST = dict(phase=0.0, run=0, tuck=0)


def sample(rng, animal=None, hd=False):
    sc = _scene(rng, animal, (0.72, 1.0))
    s = sc["scale"]
    facing = "left" if rng.integers(2) else "right"
    m = 100 * s
    cx = float(rng.uniform(m, 256 - m)) if m < 128 else 128.0
    if sc["ground"] is not None:
        cy = sc["ground"] - 80 * s
    else:
        cy = float(rng.uniform(min(128, m + 10), max(128, 256 - m - 10)))
    pose = dict(phase=float(rng.uniform(0, 6.28)), run=0, tuck=0)
    img = render(sc, pose, cx, cy, facing, shadow_y=sc["ground"])
    small, big = _outputs(img, hd)
    out = dict(image=small, animal=sc["animal"], color=sc["color"], bg=sc["bg"], facing=facing,
               scale=round(s, 3), x=round(cx / 4, 2), y=round(cy / 4, 2))
    if hd:
        out["hd"] = big
    return out


def animate(rng, action, animal=None, hd=False):
    assert action in ACTIONS, action
    sc = _scene(rng, animal, (0.55, 0.65))
    s = sc["scale"]
    ground = sc["ground"]
    base_y = (ground - 80 * s) if ground is not None else 150.0
    frames, hds = [], []
    for t in range(FRAMES):
        u = t / (FRAMES - 1)
        phase = 2 * math.pi * t / 4
        facing, sx, cx, cy = "right", 1.0, 128.0, base_y
        pose = dict(phase=phase, run=0, tuck=0)
        if action.startswith("run"):
            pose["run"] = 1
            d = 1 if action == "run_right" else -1
            facing = "right" if d > 0 else "left"
            cx = 128 + d * (-70 + 140 * u)
            cy = base_y - abs(math.sin(phase / 2 * 2)) * 6 * s
        elif action == "jump":
            pose["tuck"] = 1 if 0.1 < u < 0.9 else 0
            cy = base_y - 4 * 75 * u * (1 - u)
            cx = 108 + 40 * u          # 起跳向前带一点位移
            facing = "right"
        else:  # turn
            facing = "right" if t < FRAMES // 2 else "left"
            sx = max(0.12, abs(math.cos(math.pi * u)))
        img = render(sc, pose, cx, cy, facing, sx=sx, shadow_y=ground)
        small, big = _outputs(img, hd)
        frames.append(small)
        if hd:
            hds.append(big)
    out = dict(frames=frames, animal=sc["animal"], color=sc["color"], bg=sc["bg"], action=action, scale=round(s, 3))
    if hd:
        out["hd_frames"] = hds
    return out


# ---------------------------------------------------------------- 出样

def _font(size):
    from PIL import ImageFont
    import os
    for p in (os.path.expanduser("~/.fonts/NotoSansSC-Bold.otf"), "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _sheet(rows, tile, label_w):
    n = len(rows[0][1])
    pad = tile // 16
    sheet = Image.new("RGB", (label_w + n * (tile + pad) + pad, len(rows) * (tile + pad) + pad), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = _font(tile // 5)
    for r, (name, imgs) in enumerate(rows):
        y = pad + r * (tile + pad)
        d.text((pad * 2, y + tile // 2), name, fill=(40, 40, 40), font=f, anchor="lm")
        for c, im in enumerate(imgs):
            sheet.paste(im, (label_w + pad + c * (tile + pad), y))
    return sheet


def main():
    import argparse
    import pathlib
    import time
    ap = argparse.ArgumentParser(description="生成卡通小世界样张:sheet.png、sheet_hd.png、4 段 gif,并测速")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=500, help="测速张数")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows_lo, rows_hd = [], []
    for an in ANIMALS:
        ss = [sample(rng, an, hd=True) for _ in range(8)]
        rows_lo.append((ANIMAL_ZH[an], [Image.fromarray(x["image"]).resize((256, 256), Image.NEAREST) for x in ss]))
        rows_hd.append((ANIMAL_ZH[an], [Image.fromarray(x["hd"]) for x in ss]))
    _sheet(rows_lo, 256, 200).save(out / "sheet.png")
    _sheet(rows_hd, 256, 200).save(out / "sheet_hd.png")
    for i, (action, an) in enumerate(zip(ACTIONS, ["cat", "dog", "rabbit", "bear"])):
        r = animate(np.random.default_rng(a.seed + 1 + i), action, an, hd=True)
        ims = [Image.fromarray(f) for f in r["hd_frames"]]
        ims[0].save(out / f"{action}_{an}.gif", save_all=True, append_images=ims[1:], duration=90, loop=0)
    t0 = time.perf_counter()
    rb = np.random.default_rng(123)
    for _ in range(a.bench):
        sample(rb)
    dt = (time.perf_counter() - t0) / a.bench
    print(f"sample: {dt * 1000:.1f} ms/张,单核 1 万张约 {dt * 1e4 / 60:.1f} 分钟")
    assert np.array_equal(sample(np.random.default_rng(7))["image"], sample(np.random.default_rng(7))["image"])
    print("可复现:同种子两次输出一致")


if __name__ == "__main__":
    main()

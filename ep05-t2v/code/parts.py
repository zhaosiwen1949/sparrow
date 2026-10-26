"""小麻雀自己的小零件:词表、判卷用的两个小分类器(图、视频)、数据读取。

预训练的冻结零件在 frozen.py;判卷分类器由 train_judges.py 训,只用真数据,和主模型无关。
"""
import torch
import torch.nn as nn

# ---------------------------------------------------------------- 词表:颜色、动物、动作 + 几个记号

COLOR_ZH = {"red": "红色", "orange": "橙色", "yellow": "黄色", "green": "绿色",
            "blue": "蓝色", "purple": "紫色", "pink": "粉色", "brown": "棕色"}
ANIMAL_ZH = {"cat": "小猫", "dog": "小狗", "rabbit": "小兔", "bird": "小鸟", "fish": "小鱼", "bear": "小熊"}
ACTION_ZH = {"run_left": "向左跑", "run_right": "向右跑", "jump": "跳", "turn": "转身"}
COLORS, ANIMALS, ACTIONS = list(COLOR_ZH), list(ANIMAL_ZH), list(ACTION_ZH)
SPECIAL = ["<bos>", "<eos>", "<null>", "<img>", "</img>", "<aud>", "</aud>", "<vid>", "</vid>"]
VOCAB = [COLOR_ZH[c] for c in COLORS] + [ANIMAL_ZH[a] for a in ANIMALS] + [ACTION_ZH[a] for a in ACTIONS] + SPECIAL
ID = {w: i for i, w in enumerate(VOCAB)}
COLOR_ID = torch.tensor([ID[COLOR_ZH[c]] for c in COLORS])
ANIMAL_ID = torch.tensor([ID[ANIMAL_ZH[a]] for a in ANIMALS])
ACTION_ID = torch.tensor([ID[ACTION_ZH[a]] for a in ACTIONS])

# ---------------------------------------------------------------- 声音

SR = 24000                # TTS 与 Mimi 都是 24 kHz
MIMI_HOP = 1920           # Mimi 每 1920 个采样点(80 ms)一帧
AUD_FRAMES = 26           # 统一 26 帧 = 2.08 秒:最长的词也装得下,短的只在尾部补静音
CLIP = AUD_FRAMES * MIMI_HOP


# ---------------------------------------------------------------- 判卷用的小分类器(和主模型无关,只用真数据训)


def cnn(cin):
    def blk(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(cin, 32), blk(32, 64), blk(64, 128), blk(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())       # -> [B, 128]


class ImageJudge(nn.Module):
    """图 -> (动物, 颜色)。"""

    def __init__(self):
        super().__init__()
        self.f = cnn(3)
        self.animal, self.color = nn.Linear(128, len(ANIMALS)), nn.Linear(128, len(COLORS))

    def forward(self, x):
        h = self.f(x)
        return self.animal(h), self.color(h)


class VideoJudge(nn.Module):
    """16 帧 -> (动物, 颜色, 动作)。逐帧 CNN,再在时间上做两层一维卷积。"""

    def __init__(self):
        super().__init__()
        self.f = cnn(3)
        self.t = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, 128, 3, padding=1), nn.ReLU())
        self.animal, self.color, self.action = (nn.Linear(128, len(ANIMALS)), nn.Linear(128, len(COLORS)),
                                                nn.Linear(256, len(ACTIONS)))

    def forward(self, v):                                      # [B, T, 3, 64, 64]
        B, T = v.shape[:2]
        h = self.f(v.flatten(0, 1)).view(B, T, -1)
        s = h.mean(1)
        ht = self.t(h.transpose(1, 2))                         # [B, 128, T]
        a = torch.cat([ht.mean(2), (ht[:, :, T // 2:].mean(2) - ht[:, :, :T // 2].mean(2))], 1)
        return self.animal(s), self.color(s), self.action(a)


# ---------------------------------------------------------------- 数据


def to_float(u8):
    """uint8 [..., H, W, 3] -> float [..., 3, H, W] in [-1, 1]。"""
    return u8.movedim(-1, -3).float() / 127.5 - 1


def tail_room(wav):
    """每条录音尾部还有多少个采样点的静音(往后挪不超过它,就不会截掉字)。"""
    loud = wav.abs() > 1e-4
    last = wav.shape[-1] - 1 - torch.flip(loud, [-1]).float().argmax(-1)
    return wav.shape[-1] - 1 - last


def fit_clip(y):
    """一条 24 kHz 录音 -> (去首尾静音、前留 50 ms、尾部补静音到统一 2.08 秒的录音, 去静音后几秒);
    去静音后装不下的,录音给 None(绝不截字)。"""
    import librosa
    import numpy as np
    y, _ = librosa.effects.trim(y, top_db=40, frame_length=512, hop_length=128)
    lead, sec = SR // 20, len(y) / SR
    if lead + len(y) > CLIP:
        return None, sec
    return np.pad(y / (np.abs(y).max() + 1e-8) * 0.9, (lead, CLIP - lead - len(y))).astype(np.float32), sec


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().movedim(-3, -1)


def load_judge(cls, path, dev):
    m = cls().to(dev)
    m.load_state_dict(torch.load(path, map_location=dev)["model"])
    return m.eval().requires_grad_(False)

"""语音识别判卷:把录音转成文字,看念的是不是那个词——判的是人能不能听懂。

两个互相独立的识别器:
  - Asr(Whisper-large-v3-turbo):voice/speak.py 合成时当场听一遍、念错就重来(筛子)。
  - Paraformer(FunASR Paraformer-zh,中文专用):check_voice.py 最终把关、sparrow.py 判模型念的名字(裁判)。
    筛子和裁判不能是同一个模型,否则「过关」只说明筛子自己认得。
  - 先去掉首尾静音、两头各留 0.2 秒再送进去:Whisper 遇到大段静音会「脑补」出「谢谢观看」之类的话。
  - 集束搜索、最多 16 个 token、不许重复,防止念成一长串。
  - 「听对」按读音算:转写的拼音(不计声调)和那个词一样就算对,同音字(比如「小胸」)不算错;另报逐字全对的比例。
"""
import re

import numpy as np
import torch

REPO = "openai/whisper-large-v3-turbo"
TRAD = str.maketrans("貓魚鳥", "猫鱼鸟")          # Whisper 偶尔输出繁体
GEN = {"language": "zh", "task": "transcribe", "num_beams": 5, "max_new_tokens": 16, "no_repeat_ngram_size": 2}


def hanzi(text):
    return re.sub(r"[^一-鿿]", "", text).translate(TRAD)


def sound(text):
    from pypinyin import lazy_pinyin
    return " ".join(lazy_pinyin(hanzi(text)))


def prep(wavs, sr):
    """-> 16 kHz、去首尾静音、两头各留 0.2 秒的 numpy 列表。"""
    import librosa
    wavs = wavs.float().cpu().numpy() if torch.is_tensor(wavs) else np.asarray(wavs, dtype=np.float32)
    return [np.pad(librosa.effects.trim(librosa.resample(w, orig_sr=sr, target_sr=16000), top_db=40)[0], 3200) for w in wavs]


class Judge:
    def hits(self, wavs, sr, words):
        """(读音对上, 逐字全对, 转写文字)。"""
        texts = self(wavs, sr)
        same_sound = np.array([sound(t) == sound(w) for t, w in zip(texts, words)])
        exact = np.array([t == w for t, w in zip(texts, words)])
        return same_sound, exact, texts


class Asr(Judge):
    """Whisper-large-v3-turbo(MIT,多语种)。合成时当筛子用。"""
    name = "whisper-large-v3-turbo"

    def __init__(self, dev="cuda"):
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=REPO, device=dev,
                             torch_dtype=torch.float16 if dev.startswith("cuda") else torch.float32)

    def __call__(self, wavs, sr):
        audio = [{"raw": w, "sampling_rate": 16000} for w in prep(wavs, sr)]
        return [hanzi(o["text"]) for o in self.pipe(audio, batch_size=64, generate_kwargs=GEN)]


class Paraformer(Judge):
    """FunASR Paraformer-zh(Apache-2.0,中文专用,非自回归)。和合成时的筛子无关,最终把关、判模型念的名字都用它。"""
    name = "paraformer-zh"

    def __init__(self, dev="cuda"):
        from funasr import AutoModel
        self.m = AutoModel(model="paraformer-zh", hub="hf", device=dev, disable_update=True)

    def __call__(self, wavs, sr):
        ws = prep(wavs, sr)
        out = []
        for k in range(0, len(ws), 64):
            out += [hanzi(r["text"]) for r in self.m.generate(input=ws[k:k + 64], batch_size=64)]
        return out

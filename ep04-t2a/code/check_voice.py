"""录音把关:用独立的中文识别器(Paraformer-zh)把全部 TTS 录音转写一遍,报每个词、每个音色的识别正确率;
正确率低于 95% 的音色整个剔掉(voice.npz 就地改写)。合成时当筛子的 Whisper 也再听一遍,两个通过率都报。
data.py 没收的录音(合成时试满 4 次仍没念对的、去静音后超过 2.08 秒的)按「没念对」计入该音色。
报告写 <voice.npz 同目录>/voice_check.json;合成时的首次/最终通过率从 <录音目录>/_check.jsonl 读。

  python check_voice.py <voice.npz> <录音目录> [--samples 目录]   # --samples:挑 12 条(6 词 x 2 音色)给人听
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from asr import Asr, Paraformer
from parts import ANIMAL_ZH, ANIMALS, SR

BAR = 0.95


def rate(xs):
    return round(float(np.mean(xs)), 3) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("wavdir")
    ap.add_argument("--samples")
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    words = [ANIMAL_ZH[ANIMALS[k]] for k in d["animal"]]
    voices = [str(v) for v in d["voice"]]
    judge, judge_exact, judge_text = Paraformer().hits(d["wav"], SR, words)
    sieve, _, _ = Asr().hits(d["wav"], SR, words)
    rejected = json.load(open(a.npz.replace(".npz", "_rejected.json")))
    rows = [(v, w, bool(k)) for v, w, k in zip(voices, words, judge)]
    rows += [(f.split("_")[1], f.split("_")[0], False) for f, _ in rejected]
    by = defaultdict(lambda: defaultdict(list))
    for v, w, k in rows:
        by["voice"][v].append(k)
        by["word"][w].append(k)
        by["voice_word"][f"{v} {w}"].append(k)
    voice_rate = {v: rate(x) for v, x in sorted(by["voice"].items())}
    bad = [v for v, r in voice_rate.items() if r < BAR]
    keep = ~np.isin(d["voice"], bad)
    gen = [json.loads(line) for line in open(os.path.join(a.wavdir, "_check.jsonl"), encoding="utf-8")]
    gen = list({g["key"]: g for g in gen}.values())
    kept_rows = [r for r in rows if r[0] not in bad]
    rep = {
        "clips": len(rows),
        "sieve_whisper_at_synthesis": {"first_try": rate([len(g["tries"]) == 1 and g["ok"] for g in gen]),
                                       "after_retries": rate([g["ok"] for g in gen])},
        "sieve_whisper_now": rate(list(sieve) + [False] * len(rejected)),
        "judge_paraformer": rate([r[2] for r in rows]),
        "judge_paraformer_exact_chars": rate(list(judge_exact) + [False] * len(rejected)),
        "judge_by_word": {w: rate(x) for w, x in sorted(by["word"].items())},
        "judge_by_voice": voice_rate,
        "judge_by_voice_word": {k: rate(x) for k, x in sorted(by["voice_word"].items())},
        "dropped_voices": bad,
        "after_drop": {"clips": len(kept_rows), "judge_paraformer": rate([r[2] for r in kept_rows]),
                       "by_word": {w: rate([r[2] for r in kept_rows if r[1] == w]) for w in sorted(by["word"])},
                       "train_voices": sorted({r[0] for r in kept_rows if not r[0].startswith("te")}),
                       "test_voices": sorted({r[0] for r in kept_rows if r[0].startswith("te")})},
        "judge_misheard": [(w, v, t) for w, v, t, k in zip(words, voices, judge_text, judge) if not k],
    }
    np.savez(a.npz, **{k: x[keep] for k, x in d.items()})
    json.dump(rep, open(os.path.join(os.path.dirname(a.npz), "voice_check.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("judge_misheard", "judge_by_voice_word")}, ensure_ascii=False, indent=1))
    if a.samples:
        pick_samples(a, d, judge, judge_text, bad)


def pick_samples(a, d, judge, texts, bad):
    """6 个词 x 2 条:每个词一条训练音色、一条测试音色,男女轮换;只挑裁判听对的,拷原始录音(不补静音)。"""
    os.makedirs(a.samples, exist_ok=True)
    sex = lambda v: {"sftF": "f", "sftM": "m"}.get(v, v[-1])  # noqa: E731
    lines = []
    for i, w in enumerate(ANIMAL_ZH[x] for x in ANIMALS):
        for test, want in ((False, "fm"[i % 2]), (True, "mf"[i % 2])):
            rows = [j for j, (n, v, k) in enumerate(zip(d["name"], d["voice"], judge))
                    if k and str(v) not in bad and str(n).startswith(w + "_") and str(v).startswith("te") == test]
            rows = [j for j in rows if sex(str(d["voice"][j])) == want] or rows
            if not rows:
                continue
            j = rows[(i * 5) % len(rows)]
            v, kind = str(d["voice"][j]), "测试音色" if test else "训练音色"
            dst = f"{w}_{kind}_{'女' if sex(v) == 'f' else '男'}_{v}.wav"
            shutil.copy(os.path.join(a.wavdir, str(d["name"][j])), os.path.join(a.samples, dst))
            lines.append(f"{dst}\t原文件 {d['name'][j]}\tParaformer 听写:{texts[j]}")
    open(os.path.join(a.samples, "说明.txt"), "w", encoding="utf-8").write(
        "测试音色 = 训练从没听过的说话人(FLEURS test 份)。\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

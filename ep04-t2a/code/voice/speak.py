"""用开源 TTS 念动物名:很多个自然的真人音色,不做任何变调。

音色从哪来(都许可清楚):
  - 零样本克隆:CosyVoice2-0.5B(Apache-2.0)照着一段真人朗读学音色。参考音取自 FLEURS 普通话(CC-BY-4.0)。
    FLEURS 各份说话人不重叠:train/dev 里挑训练音色,test 里挑测试音色(训练从没听过)。
    挑的时候用 CosyVoice 自带的说话人向量(CAM++)去重,和已选的太像就跳过,保证每个音色听得出差别。
  - 内置说话人:CosyVoice-300M-SFT(Apache-2.0)自带的「中文女」「中文男」,归训练音色。
每个音色 × 3 种语速(0.9/1.0/1.1,TTS 自己的语速参数,不改音高)× 2 次随机采样。

在 TTS 容器里跑(只用 CPU;fetch.py 先把 FLEURS 和 SFT 模型下好):
  python3 speak.py pick --src <fetch 目录> --out <目录> [--train 16 --test 4]   # 选参考音 -> <目录>/voices.json、prompts/
  python3 speak.py say  --src <fetch 目录> --out <目录> --words 小猫,小狗 --shard 0 --nshards 12
每条写 <目录>/wav/<词>_<音色>_<语速>_<第几次>.wav(24 kHz,原样,不裁);已存在的跳过,可断点续跑。
合成时当场用语音识别(../asr.py,Whisper-large-v3-turbo,CPU)听写,念错就换种子重来,见 say()。
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import zlib

sys.path.append("/cosy/CosyVoice")
sys.path.append("/cosy/CosyVoice/third_party/Matcha-TTS")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SPEEDS = [0.9, 1.0, 1.1]
REPS = 2
COSY2 = "/cosy/models/CosyVoice2-0.5B"
FLEURS = "fleurs/data/cmn_hans_cn"
PLAIN = re.compile(r"[一-鿿，。、；：？！]+")        # 参考音原文只要纯汉字:夹英文、数字、括号的,克隆时容易把原文尾巴念出来
TRIES = 4                                         # 一条念错(听写不对或拖得太长)就换种子重来,最多 4 次
MAX_SEC = 1.95                                    # 去掉首尾静音后超过它,多半念了多余的东西


def pick(a):
    """从 FLEURS 挑参考音:3~6 秒、原文纯汉字、男女各半、说话人向量两两不太像。"""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    m = CosyVoice2(COSY2, load_jit=False, load_trt=False, fp16=False)
    os.makedirs(f"{a.out}/prompts", exist_ok=True)
    voices = [{"name": "sftF", "sft": "中文女"}, {"name": "sftM", "sft": "中文男"}]
    for splits, n, tag in ((("train", "dev"), a.train, "tr"), (("test",), a.test, "te")):
        rows, audio = [], {}
        for split in splits:
            rs = [r + [split] for r in csv.reader(open(f"{a.src}/{FLEURS}/{split}.tsv", encoding="utf-8"), delimiter="\t")
                  if 48000 <= int(r[5]) <= 96000 and PLAIN.fullmatch(r[2])]
            want = {r[1] for r in rs}
            with tarfile.open(f"{a.src}/{FLEURS}/audio/{split}.tar.gz", "r|gz") as tar:    # 顺序读,不回跳
                for mem in tar:
                    if os.path.basename(mem.name) in want:
                        raw = io.BytesIO(tar.extractfile(mem).read())
                        audio[os.path.basename(mem.name)] = sf.read(raw)[0].astype(np.float32)
            rows += rs
        np.random.default_rng(0).shuffle(rows)
        chosen, embs = {"FEMALE": 0, "MALE": 0}, []
        for r in rows:
            g = r[6]
            if chosen[g] >= n // 2 or r[1] not in audio:
                continue
            tmp = f"{a.out}/prompts/_candidate.wav"
            sf.write(tmp, audio[r[1]], 16000)
            e = m.frontend._extract_spk_embedding(tmp)[0]
            os.remove(tmp)
            e = e / e.norm()
            if embs and max(float(e @ x) for x in embs) > 0.7:       # 和已选的某个音色太像(同一人通常 >0.8):跳过
                continue
            embs.append(e)
            chosen[g] += 1
            name = f"{tag}{sum(chosen.values()):02d}{g[0].lower()}"
            sf.write(f"{a.out}/prompts/{name}.wav", audio[r[1]], 16000)
            voices.append({"name": name, "prompt": f"{a.out}/prompts/{name}.wav", "text": r[2], "fleurs": f"{r[7]}/{r[1]}"})
            if sum(chosen.values()) == n:
                break
        print(tag, chosen, flush=True)
    json.dump(voices, open(f"{a.out}/voices.json", "w"), ensure_ascii=False, indent=1)


def say(a):
    """合成一个分片。每条合成完当场用语音识别听一遍:听写逐字对上、且不拖长才收下,否则换种子重来。
    每条的尝试次数与听写结果记进 <目录>/check_<分片>.jsonl;试满仍不对的存成 <词>_<音色>_..._fail.wav(不进训练)。"""
    import librosa
    from asr import Asr
    voices = json.load(open(f"{a.out}/voices.json", encoding="utf-8"))
    jobs = [(w, v, sp, r) for w in a.words.split(",") for v in voices for sp in SPEEDS for r in range(REPS)]
    done = lambda j: any(os.path.exists(f"{a.out}/wav/{j[0]}_{j[1]['name']}_{j[2]}_{j[3]}{x}.wav") for x in ("", "_fail"))  # noqa: E731
    todo = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard and not done(j)]
    if not todo:
        return
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    cosy2 = CosyVoice2(COSY2, load_jit=False, load_trt=False, fp16=False)
    sft = CosyVoice(f"{a.src}/CosyVoice-300M-SFT", load_jit=False, load_trt=False, fp16=False) if any("sft" in j[1] for j in todo) else None
    asr = Asr(dev="cpu")
    os.makedirs(f"{a.out}/wav", exist_ok=True)
    log = open(f"{a.out}/check_{a.shard}.jsonl", "a", encoding="utf-8")
    for word, v, speed, rep in todo:
        key = f"{word}_{v['name']}_{speed}_{rep}"
        tries = []
        for t in range(TRIES):
            torch.manual_seed(zlib.crc32(key.encode()) + t)
            text = word + "。"
            if "sft" in v:
                m, gen = sft, sft.inference_sft(text, v["sft"], stream=False, speed=speed)
            else:
                m, gen = cosy2, cosy2.inference_zero_shot(text, v["text"], v["prompt"], stream=False, speed=speed)
            wav = torch.cat([o["tts_speech"] for o in gen], dim=1)[0].numpy()
            sec = len(librosa.effects.trim(wav, top_db=40)[0]) / m.sample_rate
            heard = asr([wav], m.sample_rate)[0] if sec <= MAX_SEC else ""
            tries.append({"sec": round(sec, 2), "heard": heard})
            if heard == word:
                break
        ok = tries[-1]["heard"] == word
        dst = f"{a.out}/wav/{key}{'' if ok else '_fail'}.wav"
        sf.write(dst + ".part.wav", wav, m.sample_rate)
        os.replace(dst + ".part.wav", dst)
        log.write(json.dumps({"key": key, "ok": ok, "tries": tries}, ensure_ascii=False) + "\n")
        log.flush()
        print(key, ok, len(tries), tries[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pick", "say"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--test", type=int, default=4)
    ap.add_argument("--words", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    pick(a) if a.what == "pick" else say(a)


if __name__ == "__main__":
    main()

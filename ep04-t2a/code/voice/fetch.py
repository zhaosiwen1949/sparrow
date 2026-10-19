"""下载造录音要用的东西(在 TTS 容器里跑,需要能上 Hugging Face):

  - FLEURS 普通话(google/fleurs 的 cmn_hans_cn,CC-BY-4.0)train/dev/test 三份(共 2.5 GB):真人朗读,用作零样本克隆的参考音。
    FLEURS 各份之间说话人不重叠,所以 train、dev 里挑的是训练音色,test 里挑的是测试音色。
  - CosyVoice-300M-SFT(Apache-2.0):自带「中文女」「中文男」两个内置说话人。
  - Whisper-large-v3-turbo(MIT):合成时当场听写把关,存进 HF 缓存(run.sh 把 HF_HOME 指到 <目录>/hf)。

  HF_HOME=<目录>/hf python3 fetch.py <目录>
"""
import sys

from huggingface_hub import hf_hub_download, snapshot_download

out = sys.argv[1]
for f in ["train.tsv", "dev.tsv", "test.tsv", "audio/train.tar.gz", "audio/dev.tar.gz", "audio/test.tar.gz"]:
    print(hf_hub_download("google/fleurs", f"data/cmn_hans_cn/{f}", repo_type="dataset", local_dir=f"{out}/fleurs"), flush=True)
print(snapshot_download("FunAudioLLM/CosyVoice-300M-SFT", local_dir=f"{out}/CosyVoice-300M-SFT",
                        ignore_patterns=[".DS_Store", "asset/*"]), flush=True)
print(snapshot_download("openai/whisper-large-v3-turbo", allow_patterns=["*.json", "*.safetensors", "*.txt"]), flush=True)

#!/bin/bash
# 在 TTS 机上造录音(只用 CPU,不挂 GPU)。
#   run.sh <CosyVoice 镜像> <CosyVoice 目录> <本目录> fetch|say [词,词,... 分片数]
#   fetch:下载 FLEURS、CosyVoice-300M-SFT、Whisper(需联网)
#   say  :第一次先挑参考音(voices.json),再后台并发合成;录音写到 <本目录>/data/wav/,日志 <本目录>/data/log_<分片>.txt
# <CosyVoice 目录> 里要有 CosyVoice/(代码)和 models/CosyVoice2-0.5B/;<本目录>/.. 是代码目录(要用其中的 asr.py)。
# 容器名 sparrow-tts-<分片>,跑完自删。
set -eu
image=$1 cosy=$2 here=$(cd "$3" && pwd) what=$4
common=(-e HF_HOME=/work/src/hf -e PYTHONPATH=/code -v "$cosy:/cosy:ro" -v "$here:/work" -v "$here/..:/code:ro")
if [ "$what" = fetch ]; then
  docker run --rm -e http_proxy -e https_proxy -e HF_HUB_DISABLE_XET=1 "${common[@]}" "$image" python3 /work/fetch.py /work/src
  exit
fi
words=$5 n=${6:-12}
mkdir -p "$here/data"
if [ ! -f "$here/data/voices.json" ]; then
  docker run --rm -e THREADS=16 "${common[@]}" "$image" \
    python3 /work/speak.py pick --src /work/src --out /work/data > "$here/data/log_pick.txt" 2>&1
fi
for s in $(seq 0 $((n - 1))); do
  docker run --rm -d --name "sparrow-tts-$s" -e THREADS=8 -e HF_HUB_OFFLINE=1 "${common[@]}" "$image" \
    bash -c "python3 /work/speak.py say --src /work/src --out /work/data --words $words --shard $s --nshards $n > /work/data/log_$s.txt 2>&1"
done

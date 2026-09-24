#!/bin/bash
# 下载 Qwen3-Embedding-0.6B 到 models/（ModelScope，8 路并行分片）
#
# 为什么不用 huggingface / hf-mirror（实测）：
#   · 直连 hf-mirror：7.5 KB/s —— 1.14GB 要 40+ 小时
#   · 单连接续传：~150 KB/s，且频繁 SSL: UNEXPECTED_EOF_WHILE_READING
#   · huggingface.co 走 Clash 代理：拿 API 元数据可以，拿 LFS 文件超时
#   · ModelScope：842 KB/s，再加 8 路并行 → 十几分钟
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/Qwen3-Embedding-0.6B"
REPO="https://modelscope.cn/api/v1/models/Qwen/Qwen3-Embedding-0.6B/repo?Revision=master&FilePath="
mkdir -p "$DIR"; cd "$DIR"

# 小文件直接下
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do
  [ -s "$f" ] || curl -sL --retry 3 -o "$f" "$REPO$f"
done

# 大文件：8 路 Range 并行
SIZE=1191586416; N=8; CH=$(( SIZE / N ))
if [ ! -s model.safetensors ] || [ "$(stat -f%z model.safetensors)" -ne "$SIZE" ]; then
  echo "并行下载权重（$(( SIZE/1048576 )) MB，$N 路）…"
  for i in $(seq 0 $((N-1))); do
    START=$(( i * CH )); END=$(( i == N-1 ? SIZE-1 : START + CH - 1 ))
    ( curl -sL -r $START-$END --retry 8 --retry-all-errors --retry-delay 2 --max-time 2400 \
        -o "part$i" "$REPO"model.safetensors && echo "  part$i 完成" ) &
  done
  wait
  cat part0 part1 part2 part3 part4 part5 part6 part7 > model.safetensors
  rm -f part*
fi

SZ=$(stat -f%z model.safetensors)
if [ "$SZ" -eq "$SIZE" ]; then echo "✅ 权重就绪：$SZ 字节"; else echo "❌ 大小不符：$SZ / $SIZE"; exit 1; fi

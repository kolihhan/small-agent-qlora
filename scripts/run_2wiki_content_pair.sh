#!/usr/bin/env bash
set -euo pipefail
SHARD="$1"
test "$SHARD" -ge 0 && test "$SHARD" -le 3
mkdir -p paired
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | sudo tar --zstd -x -C /usr
ollama serve > paired/ollama.log 2>&1 &
for i in {1..60}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 1
done
ollama pull qwen3.5:9b-q4_K_M
if [ $((SHARD%2)) -eq 0 ]; then
  ORDER="native body_index"
else
  ORDER="body_index native"
fi
for arm in $ORDER; do
  for i in "$SHARD" "$((SHARD+4))"; do
    python -m scripts.eval_2wiki_content8 --cohort frozen.json \
      --index "$i" --arm "$arm" --output "paired/case-$i-$arm.json"
  done
done

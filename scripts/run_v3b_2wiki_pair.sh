#!/usr/bin/env bash
set -euo pipefail
SHARD="$1"
test "$SHARD" -ge 0 && test "$SHARD" -le 3
mkdir -p model results /tmp/qwen9b-llama
ADAPTER_SHA=e7caec609a905d21e056b107519d78916d3ef79e7bdc70db916e40d801d2187d
MODEL_SHA=8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54
LLAMA_SHA=36ca310be4405acb7ba59b32dbdce32bb9cd7b1ef95358997b7eaed2c6f0188b
echo "$ADAPTER_SHA  adapter/qwen35-9b-qlora-v3b-adapter.gguf" | sha256sum -c -
curl -fL --retry 4 'https://huggingface.co/twoody/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true' -o model/original.gguf
echo "$MODEL_SHA  model/original.gguf" | sha256sum -c -
curl -fL --retry 4 'https://github.com/ggml-org/llama.cpp/releases/download/b11541/llama-b11541-bin-ubuntu-x64.tar.gz' -o /tmp/qwen9b-llama.tar.gz
echo "$LLAMA_SHA  /tmp/qwen9b-llama.tar.gz" | sha256sum -c -
tar -xzf /tmp/qwen9b-llama.tar.gz -C /tmp/qwen9b-llama
SERVER=$(find /tmp/qwen9b-llama -type f -name llama-server -perm -111 | head -1)
test -n "$SERVER"
run_arm() {
  local arm="$1" pid
  local extra=()
  if [ "$arm" = v3b ]; then
    extra=(--lora adapter/qwen35-9b-qlora-v3b-adapter.gguf)
  fi
  "$SERVER" -m model/original.gguf "${extra[@]}" --alias qwen3.5-9b \
    --host 127.0.0.1 --port 8080 --ctx-size 8192 --threads 4 \
    --parallel 1 --jinja --reasoning-budget 0 --temperature 0 \
    > "results/server-$arm.log" 2>&1 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null || true' RETURN
  for i in {1..180}; do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8080/health
  for i in "$SHARD" "$((SHARD+4))"; do
    python -m scripts.eval_2wiki_v3b_novel8 --cohort frozen.json \
      --index "$i" --arm "$arm" --output "results/case-$i-$arm.json"
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - RETURN
  sleep 2
}
if [ $((SHARD%2)) -eq 0 ]; then
  run_arm base
  run_arm v3b
else
  run_arm v3b
  run_arm base
fi

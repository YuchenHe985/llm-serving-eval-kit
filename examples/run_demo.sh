#!/usr/bin/env bash
# Simulated demo: two fake servers (different capacity) -> bench -> report -> plot.
# The fake server is a timing model; nothing here measures a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
OUT=examples/demo
mkdir -p "$OUT"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

python3 -m llmeval.fakeserver --port 9010 --ttft 0.05 --tpot 0.01 --capacity 4  >/dev/null & pids+=($!)
python3 -m llmeval.fakeserver --port 9011 --ttft 0.05 --tpot 0.01 --capacity 16 >/dev/null & pids+=($!)
sleep 1

for spec in "9010:fake-4-slots:1.2:4" "9011:fake-16-slots:2.5:16"; do
  IFS=: read -r port label price slots <<<"$spec"
  cat > "$OUT/$label.config.json" <<JSON
{"label": "$label", "endpoint": "http://127.0.0.1:$port", "model": "fake",
 "metadata": {"gpu_model": "simulated", "gpu_count": 1, "gpu_hourly_usd": $price, "engine": "fake-server", "parallelism": "$slots slots"},
 "matrix": {"concurrency": [1, 4, 8, 16], "input_words": [32], "max_tokens": [16]},
 "repetitions": 5, "requests_per_slot": 4, "warmup_requests": 4, "timeout_s": 30}
JSON
  python3 -m llmeval bench --config "$OUT/$label.config.json" --out "$OUT/$label.json"
done

python3 -m llmeval report "$OUT/fake-4-slots.json" "$OUT/fake-16-slots.json" \
  --slo-p95-ms 400 --out "$OUT/report.md" --plot "$OUT/latency_p95.png"

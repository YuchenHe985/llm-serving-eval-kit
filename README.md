# llm-serving-eval-kit

[![ci](https://github.com/YuchenHe985/llm-serving-eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/YuchenHe985/llm-serving-eval-kit/actions/workflows/ci.yml)

Sizing, failure diagnosis and reproducible benchmarking for LLM inference deployments. It is the
tooling I wanted while evaluating SGLang on 4x RTX 4090 (PCIe) and 4x A100 (NVLink) machines: work
out how many GPUs a model needs, turn a wall of startup errors into causes and fixes, benchmark with
repetitions and confidence intervals instead of one run, and compare two setups without pretending a
ratio is caused by the one factor you care about.

Python 3.10+, standard library only (matplotlib is optional, for plots). 38 unit and end-to-end tests run in about 6 seconds.

```
llmeval size     how many GPUs, at what tensor parallelism, does this model need?
llmeval topo     read `nvidia-smi topo -m` and say what it means for TP / EP
llmeval doctor   match logs against a catalog of real startup failures -> cause and fix
llmeval bench    run a concurrency x input x output matrix with repetitions and 95% intervals
llmeval report   markdown report: latency, TTFT, TPOT, tokens/s, SLO pass/fail, $ per 1M tokens
llmeval compare  compare two runs and list every setup factor that differs
```

## Why

Choosing GPUs for LLM serving usually goes wrong in three places: nobody checks whether the model
fits before renting a box, startup failures are debugged from memory, and the benchmark that decides
the purchase is a single run on two machines that differ in five ways. Each command targets one of
those. The failure catalog and the comparison rules come from problems I hit in real runs, not from
a list I invented.

## Examples

### Size a model (planning estimate, not a measurement)

```
$ llmeval size --model qwen2.5-32b --concurrency 20 --context 512
qwen2.5-32b: 60.5 GiB weights (bf16), 256 KiB KV per token; target 20 sequences x 512 tokens

GPU                TP  GPUs  weights/GPU   KV/GPU   usable  max seqs  note
A100-SXM4-80GB      1     1      60.5 GiB   2.5 GiB  70.5 GiB        79  fits
...
A100-SXM4-40GB      2     2      30.3 GiB   1.2 GiB  34.5 GiB        67  fits
RTX 4090 24GB       4     4      15.1 GiB   0.6 GiB  20.1 GiB       158  fits, but tensor parallelism over PCIe-only GPUs pays a per-layer all-reduce cost
```

This agrees with what actually ran: a 32B model does not fit on one 24 GB or one 40 GB card, and
needed TP=4 on the RTX 4090 machine (`tests/test_sizing_topo_doctor.py` encodes those cases).
Weights are `params x bytes`; KV per token is `layers x (2 x kv_heads x head_dim) x bytes` (or the
latent size for MLA models), sharded across tensor-parallel ranks up to the KV head count. Usable
memory is `mem_fraction x capacity - overhead`; the 1.5 GiB overhead is an explicit assumption you can
change with `--overhead-gib`. Model presets are published architectures: check them against your
checkpoint's `config.json` (`--hf-config config.json --params-b N` reads it directly).

### Diagnose a failed startup

```
$ llmeval doctor startup.log
[ERROR] Container image needs a newer CUDA than the driver supports
  evidence: nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0, ...
  cause:    The image was built for a newer CUDA runtime than the host driver can run ...
  fix:      Pin an image tag built for your driver's CUDA (see `CUDA Version` in nvidia-smi), or update the driver.
[ERROR] SGLang no longer accepts --enable-prefix-caching  ...
[ERROR] NCCL failed to initialise the multi-GPU communicator  ...
```

Eleven signatures are in the catalog, [docs/failure-catalog.md](docs/failure-catalog.md) (generated from the code, checked in CI).
`llmeval doctor --nvidia-smi smi.txt --image-cuda 13.0` runs the driver-vs-image check before you start a container.

### Read the interconnect

```
$ llmeval topo tests/fixtures/topo_rtx4090_pcie.txt
4 GPUs, interconnect: pcie
- No NVLink between GPUs: all-reduce (TP) and all-to-all (EP) go over PCIe.
- Prefer data parallelism (one replica per GPU) for models that fit on one card.
- If NCCL fails to initialise, try NCCL_P2P_DISABLE=1 (slower) and ipc: host in containers.
```

### Benchmark with repetitions

`examples/matrix.example.json` is a full matrix for a real SGLang endpoint (any OpenAI-compatible
streaming server works). Each cell is warmed up, then measured `repetitions` times; results are JSON
with the environment (tool version, host, server info, your setup metadata) stored next to the numbers.

```
$ llmeval bench --config examples/matrix.example.json --out results/a100-tp4.json
$ llmeval report results/a100-tp4.json results/4090-tp4.json --slo-p95-ms 4000 \
      --gpu-count 4 --gpu-hourly-usd 1.5 --plot p95.png
```

Reports show means with 95% bootstrap intervals over repetitions, TTFT, time per output token,
output tokens/s, SLO pass/fail and dollars per million output tokens, and rank the cheapest cell that meets the SLO.
[examples/demo/](examples/demo/) holds a report and plot made against a **simulated** server, to show the format.

### Compare two runs without over-claiming

Running `llmeval compare` on my own single-run 4090 and A100 results ([docs/real-gpu-compare.md](docs/real-gpu-compare.md)):

```
## 4090_tp4  vs  a100_tp4
CONFOUNDED: 6 setup factors differ (gpu_model, interconnect, engine_version, driver, deployment, nccl_p2p_disabled).
The ratios below cannot be attributed to any single one of them.
| c=20 | latency_p50_ms | 4836 | 2754 | 1.76x | n/a (single run) |
```

The A100 machine was 1.76x faster on tensor parallelism, but the 4090 run also had NCCL P2P disabled and used a
different SGLang version, driver and deployment. The tool reports the ratio and says what it cannot be blamed on.
With one differing factor it says so and attributes the difference to it; with confidence intervals it says when two
runs are within noise.

## Design notes

- **Standard library only.** `bench` uses `http.client` and threads, so it runs anywhere Python does, including a bare GPU box.
- **A failed request is never hidden.** A request counts as successful only if it returned 200, streamed at least one token,
  ended with `data: [DONE]` and carried no error event. Latency statistics use successful requests; failures are reported separately.
- **Prefix reuse is controlled.** Requests share a system prompt within a prefix group (`prefix_groups`), with unique user text,
  so cache-hit behaviour is part of the setup instead of an accident.
- **Every output is testable.** The load generator is tested end to end against `llmeval.fakeserver`, an in-process
  OpenAI-compatible server with configurable timing and fault modes (503s, missing `[DONE]`, error events). I also broke the
  code on purpose (comparison ignoring factors, KV cache ignoring TP sharding, accepting a stream without `[DONE]`,
  dropping the NCCL signature) and checked that tests fail for each.

## Scope

- Sizing is an estimate. Engines add activation, CUDA-graph and communication buffers that vary by version; confirm with a real launch.
- Prompt length is counted in words, not tokens (no tokenizer dependency), so it varies load rather than fixing exact token counts.
- The load generator is closed-loop (fixed concurrency); it measures sustained throughput, not response to arrival rates.
- With 5 repetitions the bootstrap intervals are rough; use more repetitions for decisions that cost money.
- The benchmark runner works against any OpenAI-compatible streaming endpoint. The demo in `examples/demo/` uses a simulated server to show the report
  format; the real 4090 and A100 results in `data/real_gpu_runs.csv` were collected by hand during my SGLang evaluation and are imported for comparison.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
make demo    # simulated demo, regenerates examples/demo/
make docs    # regenerates docs/failure-catalog.md
pip install -e '.[plot]'   # optional: installs the `llmeval` command
```

## Provenance and licence

Written from scratch; no third-party code. SGLang is referenced by name only. MIT licensed.

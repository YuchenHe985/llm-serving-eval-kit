# Methodology

## Sizing

```
weights_GiB      = params x 1e9 x bytes_per_param / 2^30
kv_bytes/token   = layers x kv_elems_per_layer x bytes_per_element
                   kv_elems_per_layer = 2 x kv_heads x head_dim           (MHA / GQA)
                                      = kv_lora_rank + qk_rope_head_dim   (MLA)
kv_per_gpu_GiB   = concurrency x context x kv_bytes/token / 2^30 / shard
                   shard = min(tp, kv_heads)   (1 for MLA)
usable_per_gpu   = mem_fraction x capacity_GiB - overhead_GiB
fits             = weights/tp + kv_per_gpu <= usable_per_gpu
```

Checks against numbers I observed:

| Quantity | Formula | Observed |
| --- | --- | --- |
| Llama-3-8B bf16 weights | 16.06 GB | about 16 GB download |
| Qwen2.5-32B bf16 weights | 65 GB | about 64 GB download |
| Qwen1.5-MoE-A2.7B bf16 weights | 28.6 GB | about 28 GB download |
| Llama-3-8B KV per token | 128 KiB | matches the published GQA layout |
| 32B on 24 GB cards | needs TP=4 | ran as TP=4 (TP=2 or 1 cannot hold the weights) |
| 8B on 24 GB cards | fits one card | ran as four single-GPU replicas |

Assumptions that are not measured: `overhead_gib` (default 1.5), that the engine can use `mem_fraction` of the
card for weights plus KV, and that KV is sharded evenly over tensor-parallel ranks. MoE models keep all experts
resident, so total parameters are used for weights.

## Benchmark design

- **Cell** = concurrency x input words x max output tokens. Each repetition uses a different seed and is warmed with
  that same seed before sending `max(concurrency x requests_per_slot, 8)` measured requests. This prevents the
  confidence interval from mixing a warm first repetition with cold later prefix groups.
- **Closed loop.** `concurrency` worker threads keep that many requests in flight. Throughput is measured, not offered.
- **Metrics per repetition:** success rate; latency P50/P95/P99 and TTFT P50/P95 over successful requests;
  mean time per output token (first-to-last token gap over tokens minus one); output tokens/s = tokens of successful
  requests / wall time; requests/s.
- **Across repetitions:** the mean and a 95% percentile-bootstrap interval of the per-repetition values (2,000 resamples, fixed seed).
  With fewer than three repetitions no interval is reported.
- **Token counts** come from the server's `usage.completion_tokens` when present (`stream_options.include_usage`),
  otherwise from the number of content chunks.
- **Prefix groups:** requests `i` and `j` share a system prompt iff `i mod prefix_groups == j mod prefix_groups`.
  User text is unique per request. This is how prefix-cache warm and cold behaviour is exercised.
- **Environment capture:** tool version, Python, host, timestamp, whitelisted fields from the server's
  `/get_server_info` when available (SGLang), and the `metadata` block from your config (GPU, interconnect, engine version,
  driver, deployment, parallelism, precision, NCCL settings, memory fraction).

## Comparison rules

`compare` treats these metadata keys as possible confounders: `gpu_model`, `interconnect`, `engine`, `engine_version`,
`driver`, `deployment`, `parallelism`, `model`, `precision`, `nccl_p2p_disabled`, `mem_fraction_static`.

| Differing factors | Verdict |
| --- | --- |
| none | Same recorded configuration; differences may be noise or an unrecorded variable. |
| exactly one | The result is consistent with that recorded factor, but an observational comparison does not establish causality. |
| two or more | CONFOUNDED: the ratio cannot be attributed to any one of them. |

For each aligned cell the tool prints "B better by" (A/B for latency-like metrics, B/A for throughput and success rate) and,
when both runs have intervals, whether they overlap ("within noise") or not ("distinguishable"). Hand-recorded single runs
(imported from CSV) have no intervals and are labelled "n/a (single run)".

Metadata can only detect factors you recorded. A variable nobody wrote down (a different prompt set, a noisy neighbour)
cannot be flagged, which is why the config asks for the setup fields explicitly.

## Cost

`$ per 1M output tokens = gpu_count x $/GPU-hour / (tokens_per_s x 3600) x 1e6`. It assumes the measured throughput is
sustained and the GPUs are billed for whole hours, so it ranks setups; it is not a quote.

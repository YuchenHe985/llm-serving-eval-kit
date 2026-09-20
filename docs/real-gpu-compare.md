# Comparing the real 4090 and A100 runs

Generated with `llmeval import-runs` and `llmeval compare` from [data/real_gpu_runs.csv](../data/real_gpu_runs.csv).
These are single runs recorded by hand, so there are no confidence intervals. The point of this page is the
**verdict line**: the tool lists every setup factor that differs between the two runs and refuses to attribute the
ratio to interconnect alone.

## 4090_dp4  vs  a100_dp4

**CONFOUNDED: 5 setup factors differ (gpu_model, interconnect, engine_version, driver, deployment). The ratios below cannot be attributed to any single one of them.**

| factor | A | B |
| --- | --- | --- |
| gpu_model | 4x RTX 4090 24GB | 4x A100-SXM4-40GB |
| interconnect | PCIe (PHB) | NVLink NV12 |
| engine_version | 0.5.16 (cu129 runtime image) | 0.5.15.post1 |
| driver | 575.51.03 / CUDA <=12.9 | 580.65.06 / CUDA 13.0 |
| deployment | Docker | bare metal (non-privileged container) |

| cell | metric | A | B | B better by | vs run-to-run noise |
| --- | --- | ---: | ---: | ---: | --- |
| c=40 | latency_p50_ms | 1215 | 921 | 1.32x | n/a (single run) |
| c=40 | latency_p95_ms | 2300 | 1749 | 1.32x | n/a (single run) |


## 4090_tp4  vs  a100_tp4

**CONFOUNDED: 6 setup factors differ (gpu_model, interconnect, engine_version, driver, deployment, nccl_p2p_disabled). The ratios below cannot be attributed to any single one of them.**

| factor | A | B |
| --- | --- | --- |
| gpu_model | 4x RTX 4090 24GB | 4x A100-SXM4-40GB |
| interconnect | PCIe (PHB) | NVLink NV12 |
| engine_version | 0.5.16 (cu129 runtime image) | 0.5.15.post1 |
| driver | 575.51.03 / CUDA <=12.9 | 580.65.06 / CUDA 13.0 |
| deployment | Docker | bare metal (non-privileged container) |
| nccl_p2p_disabled | yes | no |

| cell | metric | A | B | B better by | vs run-to-run noise |
| --- | --- | ---: | ---: | ---: | --- |
| c=20 | latency_p50_ms | 4836 | 2754 | 1.76x | n/a (single run) |
| c=20 | latency_p95_ms | 6475 | 3856 | 1.68x | n/a (single run) |


## 4090_ep4  vs  a100_ep4

**CONFOUNDED: 6 setup factors differ (gpu_model, interconnect, engine_version, driver, deployment, nccl_p2p_disabled). The ratios below cannot be attributed to any single one of them.**

| factor | A | B |
| --- | --- | --- |
| gpu_model | 4x RTX 4090 24GB | 4x A100-SXM4-40GB |
| interconnect | PCIe (PHB) | NVLink NV12 |
| engine_version | 0.5.16 (cu129 runtime image) | 0.5.15.post1 |
| driver | 575.51.03 / CUDA <=12.9 | 580.65.06 / CUDA 13.0 |
| deployment | Docker | bare metal (non-privileged container) |
| nccl_p2p_disabled | yes | no |

| cell | metric | A | B | B better by | vs run-to-run noise |
| --- | --- | ---: | ---: | ---: | --- |
| c=20 | latency_p50_ms | 1653 | 846 | 1.95x | n/a (single run) |
| c=20 | latency_p95_ms | 2095 | 1161 | 1.80x | n/a (single run) |



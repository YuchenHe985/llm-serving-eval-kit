# LLM serving benchmark report

## fake-4-slots

endpoint `http://127.0.0.1:9010` | model `fake` | tool llmeval 0.1.0 | run 2026-09-19T23:09:31+00:00

| setting | value |
| --- | --- |
| gpu_model | simulated |
| gpu_count | 1 |
| gpu_hourly_usd | 1.2 |
| engine | fake-server |
| parallelism | 4 slots |

Server info: `{'version': 'fake-0', 'tp_size': 1}`

| conc | in words | out tok | success | P50 ms | P95 ms | P99 ms | TTFT P95 ms | TPOT ms | out tok/s | P95 <= 400 ms | $/1M out tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 16 | 100.0% | 276 [274-279] | 289 [285-292] | 289 [285-292] | 62 [62-62] | 13.7 [13.5-13.8] | 58 [57-58] | pass | 5.79 |
| 4 | 32 | 16 | 100.0% | 276 [275-277] | 289 [287-291] | 289 [287-291] | 62 [62-63] | 13.6 [13.6-13.6] | 228 [227-230] | pass | 1.46 |
| 8 | 32 | 16 | 100.0% | 531 [528-534] | 555 [551-559] | 557 [554-562] | 112 [112-112] | 24.4 [24.2-24.5] | 237 [236-238] | FAIL | 1.41 |
| 16 | 32 | 16 | 100.0% | 951 [949-952] | 974 [973-976] | 986 [979-996] | 212 [211-212] | 41.9 [41.8-41.9] | 266 [266-267] | FAIL | 1.25 |

## fake-16-slots

endpoint `http://127.0.0.1:9011` | model `fake` | tool llmeval 0.1.0 | run 2026-09-19T23:10:01+00:00

| setting | value |
| --- | --- |
| gpu_model | simulated |
| gpu_count | 1 |
| gpu_hourly_usd | 2.5 |
| engine | fake-server |
| parallelism | 16 slots |

Server info: `{'version': 'fake-0', 'tp_size': 1}`

| conc | in words | out tok | success | P50 ms | P95 ms | P99 ms | TTFT P95 ms | TPOT ms | out tok/s | P95 <= 400 ms | $/1M out tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 16 | 100.0% | 274 [271-278] | 282 [277-288] | 282 [277-288] | 61 [61-62] | 13.5 [13.3-13.7] | 58 [58-59] | pass | 11.92 |
| 4 | 32 | 16 | 100.0% | 275 [273-276] | 288 [285-290] | 288 [285-290] | 63 [62-63] | 13.4 [13.4-13.5] | 229 [228-229] | pass | 3.04 |
| 8 | 32 | 16 | 100.0% | 273 [272-273] | 285 [284-286] | 287 [287-288] | 63 [63-65] | 13.4 [13.3-13.4] | 459 [457-460] | pass | 1.51 |
| 16 | 32 | 16 | 100.0% | 273 [271-274] | 286 [283-288] | 293 [290-295] | 64 [63-66] | 13.3 [13.3-13.4] | 913 [907-922] | pass | 0.76 |

## Cheapest cell that meets the SLO

| rank | run | concurrency | $/1M out tokens |
| --- | --- | ---: | ---: |
| 1 | fake-16-slots | 16 | 0.76 |
| 2 | fake-4-slots | 4 | 1.46 |

Cost assumes the measured tokens/s is sustained and GPUs are billed for the whole hour.

Values are means over repetitions with 95% bootstrap intervals in brackets; intervals are absent for single-run or imported data.

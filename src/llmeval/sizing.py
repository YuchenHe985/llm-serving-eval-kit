"""GPU memory sizing for LLM inference.

Model: per-GPU memory = weights / TP + KV cache for the target concurrency and
context + a fixed runtime overhead, against ``mem_fraction`` of the card. It is a
planning estimate, not a measurement: engines add activation, CUDA-graph and
communication buffers that vary with version and settings. ``overhead_gib`` is an
explicit, documented assumption you can change.

All sizes are GiB (2**30 bytes), which is what ``nvidia-smi`` reports in MiB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
GIB = 2**30


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float                 # total parameters in billions (all experts for MoE)
    layers: int
    kv_elems_per_token_per_layer: int  # 2 * kv_heads * head_dim, or kv_lora_rank + rope dim for MLA
    kv_heads: int = 8               # used to decide how KV shards across tensor-parallel ranks
    mla: bool = False               # MLA latent KV is kept whole on every TP rank


# Published architectures (verify against your checkpoint's config.json before relying on them).
PRESETS = {
    "llama-3-8b": ModelSpec("llama-3-8b", 8.03, 32, 2 * 8 * 128, kv_heads=8),
    "llama-3-70b": ModelSpec("llama-3-70b", 70.6, 80, 2 * 8 * 128, kv_heads=8),
    "qwen2.5-32b": ModelSpec("qwen2.5-32b", 32.5, 64, 2 * 8 * 128, kv_heads=8),
    "qwen1.5-moe-a2.7b": ModelSpec("qwen1.5-moe-a2.7b", 14.3, 24, 2 * 16 * 128, kv_heads=16),
}


def spec_from_hf_config(cfg: dict, params_b: float, name: str = "custom") -> ModelSpec:
    """Build a ModelSpec from a Hugging Face ``config.json`` dict.

    ``params_b`` must be supplied because config.json does not carry the parameter count.
    Handles grouped-query attention and DeepSeek-style MLA (``kv_lora_rank``).
    """
    layers = int(cfg["num_hidden_layers"])
    if "kv_lora_rank" in cfg:
        elems = int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim", 0))
        return ModelSpec(name, params_b, layers, elems, kv_heads=1, mla=True)
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
    return ModelSpec(name, params_b, layers, 2 * kv_heads * head_dim, kv_heads=kv_heads)


def weights_gib(spec: ModelSpec, dtype: str = "bf16") -> float:
    return spec.params_b * 1e9 * DTYPE_BYTES[dtype] / GIB


def kv_bytes_per_token(spec: ModelSpec, kv_dtype: str = "bf16") -> float:
    return spec.layers * spec.kv_elems_per_token_per_layer * DTYPE_BYTES[kv_dtype]


@dataclass(frozen=True)
class GpuSpec:
    name: str
    mem_gib: float
    nvlink: bool


CATALOG = [
    GpuSpec("RTX 4090 24GB", 24, False),
    GpuSpec("L40S 48GB", 48, False),
    GpuSpec("A100-SXM4-40GB", 40, True),
    GpuSpec("A100-SXM4-80GB", 80, True),
    GpuSpec("H100-SXM-80GB", 80, True),
    GpuSpec("H200 141GB", 141, True),
    GpuSpec("B200 192GB", 192, True),
]


@dataclass(frozen=True)
class Plan:
    gpu: str
    tp: int
    weights_per_gpu_gib: float
    kv_needed_per_gpu_gib: float
    usable_per_gpu_gib: float
    fits: bool
    max_concurrency: int      # concurrent sequences of `context_tokens` the KV budget can hold
    note: str


def plan(spec: ModelSpec, gpu: GpuSpec, tp: int, *, dtype: str = "bf16", kv_dtype: str = "bf16",
         concurrency: int = 32, context_tokens: int = 4096, mem_fraction: float = 0.9,
         overhead_gib: float = 1.5) -> Plan:
    if tp < 1:
        raise ValueError("tp must be >= 1")
    w = weights_gib(spec, dtype) / tp
    usable = gpu.mem_gib * mem_fraction - overhead_gib
    kv_total = concurrency * context_tokens * kv_bytes_per_token(spec, kv_dtype) / GIB
    shard = 1 if spec.mla else max(1, min(tp, spec.kv_heads))
    kv_per_gpu = kv_total / shard
    free = usable - w
    per_seq_gib = context_tokens * kv_bytes_per_token(spec, kv_dtype) / GIB / shard
    max_conc = int(math.floor(free / per_seq_gib)) if free > 0 and per_seq_gib > 0 else 0
    fits = free > 0 and kv_per_gpu <= free
    if free <= 0:
        note = "weights alone exceed the usable memory"
    elif not fits:
        note = f"weights fit but KV for {concurrency} x {context_tokens} tokens does not (max {max_conc} sequences)"
    elif tp > 1 and not gpu.nvlink:
        note = "fits, but tensor parallelism over PCIe-only GPUs pays a per-layer all-reduce cost"
    else:
        note = "fits"
    return Plan(gpu.name, tp, w, kv_per_gpu, usable, fits, max_conc, note)


def recommend(spec: ModelSpec, catalog=CATALOG, *, tps=(1, 2, 4, 8), **kw) -> list[Plan]:
    """Smallest feasible TP per GPU type, cheapest in GPU count first."""
    out = []
    for gpu in catalog:
        chosen: Optional[Plan] = None
        for tp in tps:
            p = plan(spec, gpu, tp, **kw)
            if p.fits:
                chosen = p
                break
        out.append(chosen or plan(spec, gpu, tps[-1], **kw))
    out.sort(key=lambda p: (not p.fits, p.tp))
    return out

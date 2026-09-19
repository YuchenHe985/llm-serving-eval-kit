"""Reproducible benchmark runner for OpenAI-compatible streaming chat endpoints.

A run is a matrix of cells (concurrency x input length x output length). Each cell
is warmed up, then measured for ``repetitions`` independent repetitions, so every
reported number carries a confidence interval instead of being a single run.
Results are JSON with the environment recorded next to the numbers.
"""
from __future__ import annotations

import http.client
import json
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from . import __version__
from .stats import bootstrap_ci, percentile

METRICS = [
    "success_rate", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "ttft_p50_ms", "ttft_p95_ms", "tpot_mean_ms", "output_tokens_per_s", "requests_per_s",
]
SCHEMA = "llmeval.result/1"

_WORDS = ("cache attention kernel latency shard tensor expert router prefill decode batch token stream "
          "memory bandwidth cluster replica gateway queue budget throughput").split()


@dataclass
class Sample:
    ok: bool
    status: int = 0
    latency_s: float = 0.0
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None
    out_tokens: int = 0
    error: str = ""


def _connect(base_url: str, timeout: float):
    u = urlparse(base_url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), (u.path.rstrip("/") or "")


def one_request(base_url: str, model: str, system: str, user: str, max_tokens: int,
                timeout: float = 60.0, include_usage: bool = True) -> Sample:
    """One streaming chat request. Tokens are counted from the server's usage chunk when
    present, otherwise from the number of content chunks."""
    payload = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    conn = None
    try:
        conn, prefix = _connect(base_url, timeout)
        conn.request("POST", prefix + "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            return Sample(False, resp.status, time.perf_counter() - t0, error=f"http {resp.status}")
        ttft = last = None
        chunks = 0
        usage_tokens = None
        done = errored = False
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if "error" in obj:
                errored = True
                continue
            if obj.get("usage") and obj["usage"].get("completion_tokens") is not None:
                usage_tokens = int(obj["usage"]["completion_tokens"])
            choices = obj.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                chunks += 1
                if ttft is None:
                    ttft = now - t0
                last = now
        latency = time.perf_counter() - t0
        tokens = usage_tokens if usage_tokens is not None else chunks
        tpot = (last - (t0 + ttft)) / (chunks - 1) if ttft is not None and chunks > 1 and last else None
        ok = done and not errored and ttft is not None
        err = "" if ok else ("error event" if errored else "stream ended without [DONE]" if not done else "no tokens")
        return Sample(ok, 200, latency, ttft, tpot, tokens, err)
    except (OSError, http.client.HTTPException) as e:
        return Sample(False, 0, time.perf_counter() - t0, error=type(e).__name__)
    finally:
        if conn is not None:
            conn.close()


def build_prompts(idx: int, prefix_groups: int, input_words: int, seed: int) -> tuple[str, str]:
    """System prompt shared inside a prefix group (drives prefix-cache reuse); unique user text."""
    group = idx % max(1, prefix_groups)
    grng = random.Random(f"{seed}-group-{group}")
    system = f"You are assistant {group}. " + " ".join(grng.choice(_WORDS) for _ in range(96))
    urng = random.Random(f"{seed}-req-{idx}")
    user = " ".join(urng.choice(_WORDS) for _ in range(max(1, input_words)))
    return system, user


def aggregate(samples: list[Sample], wall_s: float) -> dict:
    ok = [s for s in samples if s.ok]
    lat = sorted(s.latency_s * 1000 for s in ok)
    ttft = sorted(s.ttft_s * 1000 for s in ok if s.ttft_s is not None)
    tpots = [s.tpot_s * 1000 for s in ok if s.tpot_s is not None]
    out_tokens = sum(s.out_tokens for s in ok)
    return {
        "n": len(samples), "ok": len(ok),
        "success_rate": len(ok) / len(samples) if samples else 0.0,
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95), "latency_p99_ms": percentile(lat, 99),
        "ttft_p50_ms": percentile(ttft, 50), "ttft_p95_ms": percentile(ttft, 95),
        "tpot_mean_ms": sum(tpots) / len(tpots) if tpots else 0.0,
        "output_tokens_per_s": out_tokens / wall_s if wall_s > 0 else 0.0,
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        "wall_s": wall_s,
    }


def run_once(base_url, model, concurrency, input_words, max_tokens, n_requests, prefix_groups,
             timeout, seed, include_usage, offset=0) -> dict:
    def task(i):
        system, user = build_prompts(offset + i, prefix_groups, input_words, seed)
        return one_request(base_url, model, system, user, max_tokens, timeout, include_usage)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        samples = list(ex.map(task, range(n_requests)))
    return aggregate(samples, time.perf_counter() - t0)


def fetch_server_info(base_url: str, timeout: float = 3.0) -> Optional[dict]:
    """Best effort: SGLang exposes /get_server_info; other engines may not."""
    try:
        conn, prefix = _connect(base_url, timeout)
        conn.request("GET", prefix + "/get_server_info")
        r = conn.getresponse()
        body = r.read()
        conn.close()
        if r.status == 200:
            info = json.loads(body)
            keep = ("version", "model_path", "tp_size", "dp_size", "ep_size", "mem_fraction_static",
                    "dtype", "attention_backend", "served_model_name")
            return {k: info[k] for k in keep if k in info} if isinstance(info, dict) else None
    except (OSError, ValueError, http.client.HTTPException):
        pass
    return None


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for key in ("endpoint", "model", "matrix"):
        if key not in cfg:
            raise ValueError(f"config missing '{key}'")
    m = cfg["matrix"]
    for key in ("concurrency", "input_words", "max_tokens"):
        if not m.get(key):
            raise ValueError(f"matrix.{key} must be a non-empty list")
    cfg.setdefault("label", "run")
    cfg.setdefault("metadata", {})
    cfg.setdefault("repetitions", 5)
    cfg.setdefault("requests_per_slot", 4)
    cfg.setdefault("warmup_requests", 4)
    cfg.setdefault("prefix_groups", 4)
    cfg.setdefault("timeout_s", 60)
    cfg.setdefault("seed", 1)
    cfg.setdefault("include_usage", True)
    return cfg


def summarize_cell(reps: list[dict]) -> dict:
    out = {}
    for name in METRICS:
        vals = [r[name] for r in reps]
        out[name] = {"mean": sum(vals) / len(vals), "ci95": bootstrap_ci(vals) if len(vals) >= 3 else None,
                     "min": min(vals), "max": max(vals)}
    return out


def run_matrix(cfg: dict, progress=lambda msg: None) -> dict:
    base, model = cfg["endpoint"], cfg["model"]
    cells = []
    for conc in cfg["matrix"]["concurrency"]:
        for iw in cfg["matrix"]["input_words"]:
            for mt in cfg["matrix"]["max_tokens"]:
                n = max(conc * cfg["requests_per_slot"], 8)
                progress(f"cell concurrency={conc} input_words={iw} max_tokens={mt}: warmup + {cfg['repetitions']} x {n} requests")
                run_once(base, model, conc, iw, mt, cfg["warmup_requests"], cfg["prefix_groups"],
                         cfg["timeout_s"], cfg["seed"], cfg["include_usage"], offset=10_000)
                reps = [run_once(base, model, conc, iw, mt, n, cfg["prefix_groups"], cfg["timeout_s"],
                                 cfg["seed"] + r, cfg["include_usage"]) for r in range(cfg["repetitions"])]
                cells.append({"concurrency": conc, "input_words": iw, "max_tokens": mt,
                              "repetitions": reps, "summary": summarize_cell(reps)})
    return {
        "schema": SCHEMA, "label": cfg["label"], "tool": f"llmeval {__version__}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "host": platform.system() + " " + platform.machine(),
        "endpoint": base, "model": model, "metadata": cfg["metadata"],
        "server_info": fetch_server_info(base), "config": {k: v for k, v in cfg.items() if k != "metadata"},
        "cells": cells,
    }

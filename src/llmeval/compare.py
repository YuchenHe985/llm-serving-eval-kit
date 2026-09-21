"""Compare two benchmark results and refuse to over-claim.

Two runs that differ in more than one setup factor (engine version, driver,
deployment, interconnect, NCCL settings, ...) cannot support "factor X gave Nx".
This module lists the differing factors and labels the comparison accordingly.
"""
from __future__ import annotations

import csv
from typing import Optional

from .stats import intervals_overlap

CONFOUNDERS = ["gpu_model", "interconnect", "engine", "engine_version", "driver", "deployment",
               "parallelism", "model", "precision", "nccl_p2p_disabled", "mem_fraction_static"]
LOWER_IS_BETTER = {"latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "ttft_p50_ms", "ttft_p95_ms", "tpot_mean_ms"}
HIGHER_IS_BETTER = {"success_rate", "output_tokens_per_s", "requests_per_s"}


def differing_factors(a_meta: dict, b_meta: dict) -> list[tuple[str, object, object]]:
    out = []
    for k in CONFOUNDERS:
        av, bv = a_meta.get(k), b_meta.get(k)
        if av is None and bv is None:
            continue
        if str(av) != str(bv):
            out.append((k, av, bv))
    return out


def _cell_key(c: dict):
    return (c["concurrency"], c.get("input_words"), c.get("max_tokens"))


def _align(a: dict, b: dict):
    ac = {_cell_key(c): c for c in a["cells"]}
    bc = {_cell_key(c): c for c in b["cells"]}
    pairs = [(k, ac[k], bc[k]) for k in ac if k in bc]
    if pairs:
        return pairs
    by_conc_a = {c["concurrency"]: c for c in a["cells"]}
    return [((c["concurrency"], None, None), by_conc_a[c["concurrency"]], c)
            for c in b["cells"] if c["concurrency"] in by_conc_a]


def compare(a: dict, b: dict, metrics: Optional[list[str]] = None) -> dict:
    """Compare B against A. ``improvement`` > 1 means B is better than A on that metric."""
    metrics = metrics or ["latency_p50_ms", "latency_p95_ms", "output_tokens_per_s"]
    factors = differing_factors(a.get("metadata", {}), b.get("metadata", {}))
    rows = []
    for key, ca, cb in _align(a, b):
        for m in metrics:
            sa, sb = ca["summary"].get(m), cb["summary"].get(m)
            if not sa or not sb or not sa["mean"] or not sb["mean"]:
                continue
            imp = sa["mean"] / sb["mean"] if m in LOWER_IS_BETTER else sb["mean"] / sa["mean"]
            overlap = intervals_overlap(sa.get("ci95"), sb.get("ci95"))
            rows.append({"cell": key, "metric": m, "a": sa["mean"], "b": sb["mean"], "improvement": imp,
                         "noise": overlap})
    if not factors:
        verdict = "Same recorded configuration: observed differences may be run-to-run noise or unrecorded variables."
    elif len(factors) == 1:
        k, av, bv = factors[0]
        verdict = (f"One recorded factor differs ({k}: {av} -> {bv}); the result is consistent with that factor, "
                   "but this observational comparison does not by itself establish causality.")
    else:
        names = ", ".join(k for k, _, _ in factors)
        verdict = (f"CONFOUNDED: {len(factors)} setup factors differ ({names}). "
                   "The ratios below cannot be attributed to any single one of them.")
    return {"a": a.get("label"), "b": b.get("label"), "factors": factors, "confounded": len(factors) > 1,
            "rows": rows, "verdict": verdict}


def render(cmp: dict) -> str:
    lines = [f"## {cmp['a']}  vs  {cmp['b']}", "", f"**{cmp['verdict']}**", ""]
    if cmp["factors"]:
        lines += ["| factor | A | B |", "| --- | --- | --- |"]
        lines += [f"| {k} | {av} | {bv} |" for k, av, bv in cmp["factors"]]
        lines.append("")
    lines += ["| cell | metric | A | B | B better by | vs run-to-run noise |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for r in cmp["rows"]:
        conc, iw, mt = r["cell"]
        cell = f"c={conc}" + (f", in={iw}, out={mt}" if iw is not None else "")
        noise = "n/a (single run)" if r["noise"] is None else ("within noise" if r["noise"] else "distinguishable")
        lines.append(f"| {cell} | {r['metric']} | {r['a']:.4g} | {r['b']:.4g} | {r['improvement']:.2f}x | {noise} |")
    return "\n".join(lines) + "\n"


# ---- import of hand-recorded runs (for example the CSV in data/) -------------------------------------

_CSV_TO_META = {"platform": "gpu_model", "interconnect": "interconnect", "sglang_version": "engine_version",
                "driver_cuda": "driver", "deployment": "deployment", "parallelism": "parallelism",
                "model": "model", "nccl_p2p_disabled": "nccl_p2p_disabled"}


def import_runs_csv(path: str) -> list[dict]:
    """One result per CSV row (single run, no confidence interval), so hand-recorded studies can be compared."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ok, n = (int(x) for x in row["success"].split("/"))
            meta = {dst: row[src] for src, dst in _CSV_TO_META.items() if row.get(src)}
            meta["engine"] = "sglang"
            summary = {"success_rate": {"mean": ok / n, "ci95": None},
                       "latency_p50_ms": {"mean": float(row["p50_ms"]), "ci95": None},
                       "latency_p95_ms": {"mean": float(row["p95_ms"]), "ci95": None}}
            out.append({"schema": "llmeval.result/1", "label": row["run_id"], "imported": True,
                        "metadata": meta, "created_utc": row.get("date"),
                        "cells": [{"concurrency": int(row["concurrency"]), "input_words": None, "max_tokens": None,
                                   "repetitions": [], "summary": summary}]})
    return out

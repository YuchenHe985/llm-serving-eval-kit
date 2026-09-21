"""Command line: llmeval {size,topo,doctor,bench,import-runs,compare,report}."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import bench, compare as cmp, doctor, report, sizing, topo


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def cmd_size(a) -> int:
    if a.hf_config:
        if a.params_b is None:
            print("--params-b is required with --hf-config", file=sys.stderr)
            return 2
        spec = sizing.spec_from_hf_config(_read_json(a.hf_config), a.params_b, os.path.basename(os.path.dirname(a.hf_config)) or "custom")
    else:
        if a.model not in sizing.PRESETS:
            print(f"unknown model {a.model!r}; presets: {', '.join(sizing.PRESETS)}", file=sys.stderr)
            return 2
        spec = sizing.PRESETS[a.model]
    kw = dict(dtype=a.dtype, kv_dtype=a.kv_dtype, concurrency=a.concurrency, context_tokens=a.context,
              mem_fraction=a.mem_fraction, overhead_gib=a.overhead_gib)
    print(f"{spec.name}: {sizing.weights_gib(spec, a.dtype):.1f} GiB weights ({a.dtype}), "
          f"{sizing.kv_bytes_per_token(spec, a.kv_dtype) / 1024:.0f} KiB KV per token; "
          f"target {a.concurrency} sequences x {a.context} tokens\n")
    print(f"{'GPU':<18}{'TP':>3}{'GPUs':>6}{'weights/GPU':>13}{'KV/GPU':>9}{'usable':>9}{'max seqs':>10}  note")
    for p in sizing.recommend(spec, **kw):
        print(f"{p.gpu:<18}{p.tp:>3}{p.tp:>6}{p.weights_per_gpu_gib:>10.1f} GiB{p.kv_needed_per_gpu_gib:>6.1f} GiB"
              f"{p.usable_per_gpu_gib:>6.1f} GiB{p.max_concurrency:>10}  {p.note}")
    return 0


def cmd_topo(a) -> int:
    t = topo.parse_topo(_read(a.file))
    print(f"{len(t.gpus)} GPUs, interconnect: {t.interconnect}")
    for line in t.advice():
        print(f"- {line}")
    return 0


def cmd_doctor(a) -> int:
    findings = []
    text = "\n".join(_read(p) for p in a.logs) if a.logs else ""
    findings += doctor.diagnose(text)
    if a.nvidia_smi and a.image_cuda:
        drv = doctor.parse_driver_cuda(_read(a.nvidia_smi))
        if drv is None:
            print("could not find 'CUDA Version' in the nvidia-smi output", file=sys.stderr)
        else:
            f = doctor.check_cuda_compat(drv, float(a.image_cuda))
            if f and not any(x.id == f.id for x in findings):
                findings.insert(0, f)
    if a.topo:
        t = topo.parse_topo(_read(a.topo))
        print(f"Interconnect: {t.interconnect}. " + " ".join(t.advice()) + "\n")
    if not findings:
        print("No known failure signature found.")
        return 0
    for f in findings:
        print(f"[{f.severity.upper()}] {f.title}\n  evidence: {f.evidence}\n  cause:    {f.cause}\n  fix:      {f.fix}\n")
    return 1 if any(f.severity == "error" for f in findings) else 0


def cmd_catalog(a) -> int:
    print(doctor.catalog_markdown())
    return 0


def cmd_bench(a) -> int:
    cfg = bench.load_config(a.config)
    res = bench.run_matrix(cfg, progress=lambda m: print(m, file=sys.stderr))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    _write_json(a.out, res)
    print(f"wrote {a.out}")
    return 0


def cmd_import(a) -> int:
    os.makedirs(a.out_dir, exist_ok=True)
    for r in cmp.import_runs_csv(a.csv):
        path = os.path.join(a.out_dir, f"{r['label']}.json")
        _write_json(path, r)
        print(f"wrote {path}")
    return 0


def cmd_compare(a) -> int:
    ra, rb = (_read_json(p) for p in (a.a, a.b))
    print(cmp.render(cmp.compare(ra, rb)))
    return 0


def cmd_report(a) -> int:
    results = [_read_json(p) for p in a.results]
    md = report.render_markdown(results, a.slo_p95_ms, a.gpu_count, a.gpu_hourly_usd)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {a.out}")
    else:
        print(md)
    if a.plot:
        report.plot_latency(results, a.plot)
        print(f"wrote {a.plot}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llmeval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("size", help="estimate GPUs needed for a model")
    s.add_argument("--model", default="llama-3-8b", help="preset: " + ", ".join(sizing.PRESETS))
    s.add_argument("--hf-config"), s.add_argument("--params-b", type=float)
    s.add_argument("--dtype", default="bf16", choices=sizing.DTYPE_BYTES)
    s.add_argument("--kv-dtype", default="bf16", choices=sizing.DTYPE_BYTES)
    s.add_argument("--concurrency", type=int, default=32)
    s.add_argument("--context", type=int, default=4096)
    s.add_argument("--mem-fraction", type=float, default=0.9)
    s.add_argument("--overhead-gib", type=float, default=1.5)
    s.set_defaults(fn=cmd_size)

    s = sub.add_parser("topo", help="interpret `nvidia-smi topo -m` output"); s.add_argument("file", help="file or - for stdin"); s.set_defaults(fn=cmd_topo)

    s = sub.add_parser("doctor", help="diagnose startup failures from logs")
    s.add_argument("logs", nargs="*"), s.add_argument("--topo"), s.add_argument("--nvidia-smi"), s.add_argument("--image-cuda")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("catalog", help="print the failure-signature catalog as markdown"); s.set_defaults(fn=cmd_catalog)

    s = sub.add_parser("bench", help="run a benchmark matrix"); s.add_argument("--config", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("import-runs", help="turn a hand-recorded CSV into result files"); s.add_argument("csv"); s.add_argument("--out-dir", required=True); s.set_defaults(fn=cmd_import)

    s = sub.add_parser("compare", help="compare two result files, flagging confounders"); s.add_argument("a"); s.add_argument("b"); s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("report", help="markdown report (and optional plot)")
    s.add_argument("results", nargs="+"), s.add_argument("--slo-p95-ms", type=float), s.add_argument("--gpu-count", type=int)
    s.add_argument("--gpu-hourly-usd", type=float), s.add_argument("--out"), s.add_argument("--plot")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

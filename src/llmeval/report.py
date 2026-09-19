"""Markdown report and optional plot from benchmark results."""
from __future__ import annotations

from typing import Optional


def _fmt(s: Optional[dict], digits: int = 1) -> str:
    if not s:
        return "-"
    txt = f"{s['mean']:.{digits}f}"
    if s.get("ci95"):
        lo, hi = s["ci95"]
        txt += f" [{lo:.{digits}f}-{hi:.{digits}f}]"
    return txt


def cost_per_million_tokens(gpu_count: int, gpu_hourly_usd: float, tokens_per_s: float) -> Optional[float]:
    """USD per 1M output tokens for a deployment that sustains ``tokens_per_s`` on ``gpu_count`` GPUs."""
    if tokens_per_s <= 0:
        return None
    return gpu_count * gpu_hourly_usd / (tokens_per_s * 3600) * 1e6


def render_markdown(results: list[dict], slo_p95_ms: Optional[float] = None,
                    gpu_count: Optional[int] = None, gpu_hourly_usd: Optional[float] = None) -> str:
    lines = ["# LLM serving benchmark report", ""]
    best_rows = []
    for r in results:
        meta = r.get("metadata", {})
        lines += [f"## {r.get('label', 'run')}", ""]
        info = [f"endpoint `{r.get('endpoint', '-')}`", f"model `{r.get('model', '-')}`",
                f"tool {r.get('tool', '-')}", f"run {r.get('created_utc', '-')}"]
        lines += [" | ".join(info), ""]
        if meta:
            lines += ["| setting | value |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in meta.items()] + [""]
        if r.get("server_info"):
            lines += [f"Server info: `{r['server_info']}`", ""]
        gc = int(meta.get("gpu_count", gpu_count or 0)) or None
        price = float(meta.get("gpu_hourly_usd", gpu_hourly_usd or 0)) or None
        head = ["conc", "in words", "out tok", "success", "P50 ms", "P95 ms", "P99 ms", "TTFT P95 ms", "TPOT ms", "out tok/s"]
        if slo_p95_ms:
            head.append(f"P95 <= {slo_p95_ms:g} ms")
        if gc and price:
            head.append("$/1M out tok")
        lines += ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        for c in r["cells"]:
            s = c["summary"]
            row = [str(c["concurrency"]), str(c.get("input_words") or "-"), str(c.get("max_tokens") or "-"),
                   f"{s['success_rate']['mean'] * 100:.1f}%", _fmt(s.get("latency_p50_ms"), 0), _fmt(s.get("latency_p95_ms"), 0),
                   _fmt(s.get("latency_p99_ms"), 0), _fmt(s.get("ttft_p95_ms"), 0), _fmt(s.get("tpot_mean_ms"), 1),
                   _fmt(s.get("output_tokens_per_s"), 0)]
            meets = None
            if slo_p95_ms and s.get("latency_p95_ms"):
                meets = s["latency_p95_ms"]["mean"] <= slo_p95_ms
                row.append("pass" if meets else "FAIL")
            cost = None
            if gc and price and s.get("output_tokens_per_s"):
                cost = cost_per_million_tokens(gc, price, s["output_tokens_per_s"]["mean"])
                row.append(f"{cost:.2f}" if cost is not None else "-")
            lines.append("| " + " | ".join(row) + " |")
            if cost is not None and (meets is None or meets):
                best_rows.append((cost, r.get("label"), c["concurrency"]))
        lines.append("")
    if best_rows:
        best_rows.sort()
        lines += ["## Cheapest cell that meets the SLO" if slo_p95_ms else "## Cheapest cell", "",
                  "| rank | run | concurrency | $/1M out tokens |", "| --- | --- | ---: | ---: |"]
        seen = set()
        rank = 0
        for cost, label, conc in best_rows:
            if label in seen:
                continue
            seen.add(label)
            rank += 1
            lines.append(f"| {rank} | {label} | {conc} | {cost:.2f} |")
        lines += ["", "Cost assumes the measured tokens/s is sustained and GPUs are billed for the whole hour.", ""]
    lines += ["Values are means over repetitions with 95% bootstrap intervals in brackets; "
              "intervals are absent for single-run or imported data.", ""]
    return "\n".join(lines)


def plot_latency(results: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot (pip install matplotlib)") from e
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, r in enumerate(results):
        cells = sorted(r["cells"], key=lambda c: c["concurrency"])
        xs = [c["concurrency"] for c in cells]
        ys = [c["summary"]["latency_p95_ms"]["mean"] for c in cells]
        ax.plot(xs, ys, color=colors[i % 3], linewidth=2, marker="o", markersize=6, label=r.get("label"))
        lo = [c["summary"]["latency_p95_ms"]["ci95"][0] if c["summary"]["latency_p95_ms"].get("ci95") else y for c, y in zip(cells, ys)]
        hi = [c["summary"]["latency_p95_ms"]["ci95"][1] if c["summary"]["latency_p95_ms"].get("ci95") else y for c, y in zip(cells, ys)]
        ax.fill_between(xs, lo, hi, color=colors[i % 3], alpha=0.12, linewidth=0)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("P95 latency (ms)")
    ax.grid(True, color="#e6e5e1")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

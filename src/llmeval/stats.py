"""Small statistics helpers (standard library only)."""
from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Optional, Sequence


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of an ascending sequence (p in 0..100)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = max(0, min(n - 1, math.ceil(p / 100 * n) - 1))
    return float(sorted_vals[idx])


def summarize(vals: Sequence[float]) -> dict:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(vals)
    return {"count": len(s), "mean": statistics.fmean(s), "p50": percentile(s, 50),
            "p95": percentile(s, 95), "p99": percentile(s, 99), "max": float(s[-1])}


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] = statistics.fmean,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Optional[tuple[float, float]]:
    """Percentile bootstrap confidence interval. Returns None with fewer than 3 values."""
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def intervals_overlap(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[bool]:
    """True/False if both intervals exist, None if either is missing."""
    if a is None or b is None:
        return None
    return a[0] <= b[1] and b[0] <= a[1]

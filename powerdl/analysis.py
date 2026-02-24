from __future__ import annotations
from typing import List, Tuple
import math
from .recorder import Sample, Mark

def integrate_joules(samples: List[Sample], t0: float, t1: float) -> float:
    if t1 <= t0 or len(samples) < 2:
        return 0.0
    s = [x for x in samples if (t0 <= x.t <= t1) and (x.p_w == x.p_w)]
    if len(s) < 2:
        return 0.0
    e = 0.0
    for a, b in zip(s, s[1:]):
        dt = b.t - a.t
        if dt > 0:
            e += 0.5 * (a.p_w + b.p_w) * dt
    return float(e) if math.isfinite(e) else 0.0

def pair_marks(marks: List[Mark], start_name: str, end_name: str) -> List[Tuple[Mark, Mark]]:
    pairs = []
    pending = None
    for m in marks:
        if m.name == start_name:
            pending = m
        elif m.name == end_name and pending is not None:
            pairs.append((pending, m))
            pending = None
    return pairs

from __future__ import annotations
from typing import List, Tuple
import math
from .recorder import Mark, Sample
from .analysis import integrate_joules, pair_marks

def _avg_power_w(samples: List[Sample], t0: float, t1: float) -> float:
    e = integrate_joules(samples, t0, t1)
    dt = max(1e-9, (t1 - t0))
    return e / dt

def summarize_session(recorder, batch_size: int | None = None) -> dict:
    """
    Expects marks: train_start/train_end, epoch_start/epoch_end, batch_start/batch_end
    Optionally infer_start/infer_end
    """
    out = {}

    # total train
    total_pairs = pair_marks(recorder.marks, "train_start", "train_end")
    if total_pairs:
        a, b = total_pairs[0]
        E = integrate_joules(recorder.samples, a.t, b.t)
        out["train_energy_j"] = E
        out["train_time_s"] = (b.t - a.t)
        out["train_avg_power_w"] = _avg_power_w(recorder.samples, a.t, b.t)

    # per epoch
    epoch_pairs = pair_marks(recorder.marks, "epoch_start", "epoch_end")
    out["epoch_energies"] = []
    for a, b in epoch_pairs:
        e = integrate_joules(recorder.samples, a.t, b.t)
        out["epoch_energies"].append({
            "epoch": a.meta.get("epoch", None),
            "energy_j": e,
            "time_s": (b.t - a.t),
            "avg_power_w": _avg_power_w(recorder.samples, a.t, b.t),
        })

    # per batch/step
    batch_pairs = pair_marks(recorder.marks, "batch_start", "batch_end")
    batch_rows = []
    for a, b in batch_pairs:
        e = integrate_joules(recorder.samples, a.t, b.t)
        dt = (b.t - a.t)
        bs = b.meta.get("batch_size", None) or batch_size
        j_per_sample = (e / bs) if (bs and bs > 0) else None
        ips = (bs / dt) if (bs and dt > 0) else None
        batch_rows.append({
            "batch": b.meta.get("batch", None),
            "batch_size": bs,
            "time_s": dt,
            "energy_j": e,
            "avg_power_w": (e / dt) if dt > 0 else None,
            "j_per_sample": j_per_sample,
            "throughput_img_s": ips,
            "loss": b.meta.get("loss", None),
        })
    out["batch_rows"] = batch_rows

    # inference (optional)
    inf_pairs = pair_marks(recorder.marks, "infer_start", "infer_end")
    if inf_pairs:
        a, b = inf_pairs[-1]
        Einf = integrate_joules(recorder.samples, a.t, b.t)
        dt = (b.t - a.t)
        out["infer_energy_j"] = Einf
        out["infer_time_s"] = dt
        out["infer_avg_power_w"] = (Einf / dt) if dt > 0 else None

    return out

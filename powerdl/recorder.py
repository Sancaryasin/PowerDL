from __future__ import annotations

import os
import csv
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Sample:
    t: float
    p_w: float
    gpu_util: Optional[int] = None
    mem_util: Optional[int] = None


@dataclass
class Mark:
    name: str
    t: float
    meta: Dict[str, Any]


def _write_samples_csv(path: str, samples: List[Sample]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "p_w", "gpu_util", "mem_util"])
        w.writeheader()
        for s in samples:
            w.writerow(
                {
                    "t": float(s.t),
                    "p_w": float(s.p_w),
                    "gpu_util": s.gpu_util,
                    "mem_util": s.mem_util,
                }
            )


def _write_marks_csv(path: str, marks: List[Mark]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "t", "meta"])
        w.writeheader()
        for m in marks:
            w.writerow(
                {
                    "name": str(m.name),
                    "t": float(m.t),
                    "meta": str(m.meta or {}),
                }
            )


class PowerRecorder:
    """
    Background sampler that records power (and optionally utilization) into memory.

    Memory warning model (rough):
      estimated_mb ≈ n_samples * approx_bytes_per_sample / (1024^2)

    Auto-flush:
      When warning triggers, optionally write buffers to CSV and clear them.
    """

    def __init__(
        self,
        reader,
        interval_s: float = 0.2,
        collect_util: bool = True,
        *,
        # --- warning controls ---
        max_samples: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        approx_bytes_per_sample: int = 1024,
        warn_at_pct: float = 0.8,
        on_warning: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        # --- auto flush controls ---
        auto_flush_dir: Optional[str] = None,
        auto_flush_prefix: str = "powerdl",
        auto_flush_keep_marks: bool = True,
    ):
        self.reader = reader
        self.interval_s = float(interval_s)
        self.collect_util = bool(collect_util)

        self.samples: List[Sample] = []
        self.marks: List[Mark] = []

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

        # warning config
        self.max_samples = max_samples
        self.max_memory_mb = float(max_memory_mb) if max_memory_mb is not None else None
        self.approx_bytes_per_sample = int(approx_bytes_per_sample)
        self.warn_at_pct = float(warn_at_pct)
        self.on_warning = on_warning

        self._warned_samples = False
        self._warned_memory = False

        # auto flush config
        self.auto_flush_dir = auto_flush_dir
        self.auto_flush_prefix = str(auto_flush_prefix)
        self.auto_flush_keep_marks = bool(auto_flush_keep_marks)
        self._flush_count = 0

        # lock to protect flush/clear while sampling thread runs
        self._buf_lock = threading.Lock()

    def start(self):
        if self._th is not None and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        self._th = None

    def mark(self, name: str, **meta):
        with self._buf_lock:
            self.marks.append(Mark(name=name, t=time.time(), meta=dict(meta or {})))

    def _maybe_auto_flush(self, reason: str, payload: Dict[str, Any]) -> None:
        if not self.auto_flush_dir:
            return

        try:
            self._flush_count += 1
            tag = f"{self._flush_count:04d}"

            base = os.path.join(self.auto_flush_dir, f"{self.auto_flush_prefix}_flush_{tag}")
            samples_path = base + "_samples.csv"
            marks_path = base + "_marks.csv"

            # write snapshots
            _write_samples_csv(samples_path, self.samples)
            if self.auto_flush_keep_marks:
                _write_marks_csv(marks_path, self.marks)

            # clear buffers
            self.samples.clear()
            if not self.auto_flush_keep_marks:
                self.marks.clear()

            # reset warning flags so it can warn/flush again later
            self._warned_samples = False
            self._warned_memory = False

        except Exception as e:
            # don't crash training due to flush errors
            msg = f"[PowerDL] Auto-flush failed ({reason}): {e}"
            warnings.warn(msg, RuntimeWarning)

    def _maybe_warn_memory(self):
        n = len(self.samples)

        # sample-count limit
        if (self.max_samples is not None) and (not self._warned_samples):
            thr = int(self.max_samples * self.warn_at_pct)
            if n >= thr:
                msg = (
                    f"[PowerDL] Memory warning: sample buffer reached {n}/{self.max_samples} "
                    f"({(n/self.max_samples)*100:.1f}%). Consider increasing interval_s, "
                    f"disabling util collection, or exporting/clearing samples."
                )
                payload = {
                    "kind": "max_samples",
                    "n_samples": n,
                    "max_samples": self.max_samples,
                    "warn_at_pct": self.warn_at_pct,
                }
                warnings.warn(msg, RuntimeWarning)
                if self.on_warning:
                    try:
                        self.on_warning(msg, payload)
                    except Exception:
                        pass

                self._warned_samples = True
                # auto-flush
                self._maybe_auto_flush("max_samples", payload)

        # estimated memory limit
        if (self.max_memory_mb is not None) and (not self._warned_memory):
            est_mb = (n * self.approx_bytes_per_sample) / (1024.0 * 1024.0)
            thr_mb = self.max_memory_mb * self.warn_at_pct
            if est_mb >= thr_mb:
                msg = (
                    f"[PowerDL] Memory warning: estimated sample buffer ~{est_mb:.1f} MB "
                    f"(limit {self.max_memory_mb:.1f} MB, {(est_mb/self.max_memory_mb)*100:.1f}%). "
                    f"Consider increasing interval_s, disabling util collection, "
                    f"or exporting/clearing samples."
                )
                payload = {
                    "kind": "max_memory_mb",
                    "n_samples": n,
                    "approx_bytes_per_sample": self.approx_bytes_per_sample,
                    "estimated_mb": est_mb,
                    "max_memory_mb": self.max_memory_mb,
                    "warn_at_pct": self.warn_at_pct,
                }
                warnings.warn(msg, RuntimeWarning)
                if self.on_warning:
                    try:
                        self.on_warning(msg, payload)
                    except Exception:
                        pass

                self._warned_memory = True
                # auto-flush
                self._maybe_auto_flush("max_memory_mb", payload)

    def _run(self):
        while not self._stop.is_set():
            t = time.time()

            try:
                p = float(self.reader.read_power_watts())
            except Exception:
                p = float("nan")

            gpu_u = None
            mem_u = None
            if self.collect_util:
                try:
                    gpu_u, mem_u = self.reader.read_utilization()
                except Exception:
                    gpu_u, mem_u = None, None

            with self._buf_lock:
                self.samples.append(Sample(t=t, p_w=p, gpu_util=gpu_u, mem_util=mem_u))
                self._maybe_warn_memory()

            time.sleep(self.interval_s)
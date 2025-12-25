from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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

class PowerRecorder:
    def __init__(self, reader, interval_s: float = 0.2):
        self.reader = reader
        self.interval_s = float(interval_s)

        self.samples: List[Sample] = []
        self.marks: List[Mark] = []

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._th is not None and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)

    def mark(self, name: str, **meta) -> None:
        self.marks.append(Mark(name=name, t=time.perf_counter(), meta=dict(meta)))

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()

            p = float("nan")
            gpu_u = None
            mem_u = None

            try:
                p = float(self.reader.read_power_watts())
            except Exception:
                pass

            # utilization opsiyonel: reader destekliyorsa al
            try:
                if hasattr(self.reader, "read_utilization"):
                    gpu_u, mem_u = self.reader.read_utilization()
            except Exception:
                gpu_u, mem_u = None, None

            self.samples.append(Sample(t=t, p_w=p, gpu_util=gpu_u, mem_util=mem_u))
            time.sleep(self.interval_s)

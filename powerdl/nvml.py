from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetName,
    nvmlDeviceGetUtilizationRates,
)

@dataclass
class NvmlDevice:
    index: int = 0
    name: Optional[str] = None

class NvmlPowerReader:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._handle = None
        self.device = NvmlDevice(index=device_index)

    def __enter__(self) -> "NvmlPowerReader":
        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(self.device_index)

        name = nvmlDeviceGetName(self._handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        self.device.name = str(name)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            nvmlShutdown()
        except Exception:
            pass
        self._handle = None

    def read_power_watts(self) -> float:
        mw = nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
        return float(mw) / 1000.0

    def read_utilization(self) -> Tuple[int, int]:
        """
        Returns (gpu_util_percent, mem_util_percent)
        """
        u = nvmlDeviceGetUtilizationRates(self._handle)
        return int(u.gpu), int(u.memory)

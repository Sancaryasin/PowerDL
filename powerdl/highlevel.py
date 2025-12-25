from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Any, Dict

from .nvml import NvmlPowerReader
from .recorder import PowerRecorder
from .io import save_samples_csv, save_marks_csv, save_rows_csv
from .analysis import integrate_joules, pair_marks

# TF
from .tf import PowerCallback, summarize_energy

# Torch (opsiyonel)
try:
    from .torch import TorchRunConfig, train_one_epoch
    from .summary import summarize_session
    import torch
except Exception:
    TorchRunConfig = None
    train_one_epoch = None
    summarize_session = None
    torch = None


def _ensure_dir(out_dir: Optional[str]) -> Optional[str]:
    """Create output directory if provided. If out_dir is None, disable exports."""
    if out_dir is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _avg_util_between(samples, t0, t1, key: str):
    vals = []
    for s in samples:
        if t0 <= s.t <= t1:
            v = getattr(s, key, None)
            if v is not None:
                vals.append(v)
    return (sum(vals) / len(vals)) if vals else None


def _avg_util_for_pairs(samples, pairs, key: str):
    vals = []
    for a, b in pairs:
        v = _avg_util_between(samples, a.t, b.t, key)
        if v is not None:
            vals.append(v)
    return (sum(vals) / len(vals)) if vals else None


@dataclass
class ExportConfig:
    samples_csv: str = "samples.csv"
    marks_csv: str = "marks.csv"
    batches_csv: str = "batches.csv"
    epochs_csv: str = "epochs.csv"
    summary_json: str = "summary.json"


class PowerDLTFProfiler:
    """
    Kullanıcı kodunu sade tutar:
      - prof.fit(model, ds_or_arrays, ...)
      - prof.infer_keras(model, x, batch_size=...)
    """
    def __init__(self, out_dir: Optional[str] = "run_tf", device_index=0, interval_s=0.02, verbose: int = 0):
        self.out_dir = _ensure_dir(out_dir)
        self.device_index = device_index
        self.interval_s = float(interval_s)
        self.verbose = int(verbose)
        self.export = ExportConfig(
            samples_csv="samples_tf.csv",
            marks_csv="marks_tf.csv",
            batches_csv="batches_tf.csv",
            epochs_csv="epochs_tf.csv",
            summary_json="summary_tf.json",
        )

        self.reader = None
        self.rec = None
        self.cb = None
        self.summary: Optional[dict] = None
        self.gpu_name: Optional[str] = None

    def __enter__(self):
        self.reader = NvmlPowerReader(device_index=self.device_index).__enter__()
        self.gpu_name = self.reader.device.name

        self.rec = PowerRecorder(reader=self.reader, interval_s=self.interval_s)
        self.rec.start()
        self.rec.mark("session_start")

        self.cb = PowerCallback(self.rec)
        return self

    def __exit__(self, exc_type, exc, tb):
        # finalize + export even if exception? (minimum safe)
        try:
            self.rec.mark("session_end")
            self.rec.stop()

            # TF summary
            self.summary = summarize_energy(self.rec)
            if self.out_dir is not None:
                self._export_tf()
            if self.verbose == 1 and self.summary:
                s = self.summary
                # headline
                if s.get("train_time_s") is not None and s.get("train_energy_j") is not None:
                    print(f"[TF][TRAIN] time={s['train_time_s']:.2f}s | energy={s['train_energy_j']:.2f} J | avg_power={s.get('train_avg_power_w', float('nan')):.2f} W")
                if s.get("infer_time_s") is not None and s.get("infer_energy_j") is not None:
                    print(f"[TF][INFER] time={s['infer_time_s']:.2f}s | energy={s['infer_energy_j']:.2f} J | avg_power={s.get('infer_avg_power_w', float('nan')):.2f} W")
        finally:
            try:
                self.reader.__exit__(exc_type, exc, tb)
            except Exception:
                pass

    def fit(self, model, data, **fit_kwargs):
        """
        data: tf.data.Dataset veya (x,y) fit'e giden her şey
        fit_kwargs: epochs, verbose, batch_size, callbacks, ...
        """
        # callbacks birleştir
        user_cbs = fit_kwargs.pop("callbacks", [])
        callbacks = list(user_cbs) + [self.cb]

        # if profiler is quiet, silence keras too
        if self.verbose == 0:
            fit_kwargs["verbose"] = 0
        return model.fit(data, callbacks=callbacks, **fit_kwargs)

    def infer_keras(self, model, x, batch_size: int = 256, warmup_n: int = 1024):
        """
        Keras model.predict ölçümü: warm-up ölçüm dışı, infer ölçüm içi.
        """
        # warm-up
        self.rec.mark("warmup_start")
        _ = model.predict(x[:warmup_n], batch_size=batch_size, verbose=0)
        self.rec.mark("warmup_end")

        # measured inference
        self.rec.mark("infer_start")
        _ = model.predict(x, batch_size=batch_size, verbose=0)
        self.rec.mark("infer_end", n_samples=int(len(x)))

        # print quick inference headline
        pairs = pair_marks(self.rec.marks, "infer_start", "infer_end")
        if pairs:
            a, b = pairs[-1]
            Einf = integrate_joules(self.rec.samples, a.t, b.t)
            dt = b.t - a.t
            metrics = {
                "infer_energy_j": Einf,
                "infer_time_s": dt,
                "infer_avg_power_w": (Einf / dt) if dt > 0 else None,
                "infer_j_per_sample": Einf / len(x),
                "infer_throughput_img_s": (len(x) / dt) if dt > 0 else None,
                "infer_avg_gpu_util": _avg_util_between(self.rec.samples, a.t, b.t, "gpu_util"),
                "infer_avg_mem_util": _avg_util_between(self.rec.samples, a.t, b.t, "mem_util"),
            }
            return metrics
        return {}

    def report(self):
        """Return an in-memory report object (no file IO required)."""
        from .report import Report
        return Report(
            backend="tf",
            samples=list(self.rec.samples) if self.rec else [],
            marks=list(self.rec.marks) if self.rec else [],
            summary=dict(self.summary or {}),
        )

    def _export_tf(self):
        # raw logs
        save_samples_csv(os.path.join(self.out_dir, self.export.samples_csv), self.rec.samples)
        save_marks_csv(os.path.join(self.out_dir, self.export.marks_csv), self.rec.marks)

        # batch/epoch tables
        rows = self.summary.get("batch_rows", [])
        save_rows_csv(
            os.path.join(self.out_dir, self.export.batches_csv),
            rows,
            fieldnames=["batch","batch_size","time_s","energy_j","avg_power_w","j_per_sample","throughput_img_s","loss"],
        )
        save_rows_csv(
            os.path.join(self.out_dir, self.export.epochs_csv),
            self.summary.get("epoch_energies", []),
            fieldnames=["epoch","time_s","energy_j","avg_power_w"],
        )

        # add util rollups
        epoch_pairs = pair_marks(self.rec.marks, "epoch_start", "epoch_end")
        self.summary["epoch_avg_gpu_util_pct"] = _avg_util_for_pairs(self.rec.samples, epoch_pairs, "gpu_util")
        self.summary["epoch_avg_mem_util_pct"] = _avg_util_for_pairs(self.rec.samples, epoch_pairs, "mem_util")
        self.summary["gpu_name"] = self.gpu_name

        with open(os.path.join(self.out_dir, self.export.summary_json), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)


class PowerDLTorchProfiler:
    """
    Kullanıcı kodu sade:
      - prof.train_epochs(model, trainloader, optimizer, loss_fn, epochs=...)
      - prof.infer_tensor(model, batch_size=..., n_samples=...)
    """
    def __init__(self, out_dir: Optional[str] = "run_torch", device_index=0, interval_s=0.02, amp=True, verbose: int = 0):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch/torchvision first.")
        self.out_dir = _ensure_dir(out_dir)
        self.device_index = device_index
        self.interval_s = float(interval_s)
        self.amp = bool(amp)
        self.verbose = int(verbose)

        self.export = ExportConfig(
            samples_csv="samples_torch.csv",
            marks_csv="marks_torch.csv",
            batches_csv="batches_torch.csv",
            epochs_csv="epochs_torch.csv",
            summary_json="summary_torch.json",
        )

        self.reader = None
        self.rec = None
        self.summary: Optional[dict] = None
        self.gpu_name: Optional[str] = None
        self.cfg = None

    def __enter__(self):
        self.reader = NvmlPowerReader(device_index=self.device_index).__enter__()
        self.gpu_name = self.reader.device.name

        self.rec = PowerRecorder(reader=self.reader, interval_s=self.interval_s)
        self.rec.start()
        self.rec.mark("session_start")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = TorchRunConfig(device=device, amp=self.amp)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.rec.mark("session_end")
            self.rec.stop()

            self.summary = summarize_session(self.rec)
            if self.out_dir is not None:
                self._export_torch()
            if self.verbose == 1 and self.summary:
                s = self.summary
                if s.get("train_time_s") is not None and s.get("train_energy_j") is not None:
                    print(f"[TORCH][TRAIN] time={s['train_time_s']:.2f}s | energy={s['train_energy_j']:.2f} J | avg_power={s.get('train_avg_power_w', float('nan')):.2f} W")
                if s.get("infer_time_s") is not None and s.get("infer_energy_j") is not None:
                    print(f"[TORCH][INFER] time={s['infer_time_s']:.2f}s | energy={s['infer_energy_j']:.2f} J | avg_power={s.get('infer_avg_power_w', float('nan')):.2f} W")
        finally:
            try:
                self.reader.__exit__(exc_type, exc, tb)
            except Exception:
                pass

    def train_epochs(self, model, trainloader, optimizer, loss_fn, epochs: int):
        self.rec.mark("train_start")
        for ep in range(int(epochs)):
            train_one_epoch(model, trainloader, optimizer, loss_fn, self.rec, epoch=ep, cfg=self.cfg)
        self.rec.mark("train_end")

    # Backward-compatible aliases
    def train(self, model, trainloader, optimizer, loss_fn, epochs: int):
        return self.train_epochs(model, trainloader, optimizer, loss_fn, epochs=epochs)

    def infer(self, model, batch_size: int = 256, n_samples: int = 20000):
        return self.infer_tensor(model, batch_size=batch_size, n_samples=n_samples)

    @torch.no_grad()
    def infer_tensor(self, model, batch_size: int = 256, n_samples: int = 20000):
        device = self.cfg.device
        model.eval()
        x = torch.randn(batch_size, 3, 32, 32, device=device)

        # warm-up (out of measurement)
        self.rec.mark("warmup_start")
        for _ in range(10):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.rec.mark("warmup_end")

        iters = max(1, n_samples // batch_size)

        self.rec.mark("infer_start", n_samples=iters * batch_size)
        for _ in range(iters):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.rec.mark("infer_end", n_samples=iters * batch_size)

    def report(self):
        from .report import Report
        return Report(
            backend="torch",
            samples=list(self.rec.samples) if self.rec else [],
            marks=list(self.rec.marks) if self.rec else [],
            summary=dict(self.summary or {}),
        )

    def _export_torch(self):
        save_samples_csv(os.path.join(self.out_dir, self.export.samples_csv), self.rec.samples)
        save_marks_csv(os.path.join(self.out_dir, self.export.marks_csv), self.rec.marks)

        rows = self.summary.get("batch_rows", [])
        save_rows_csv(
            os.path.join(self.out_dir, self.export.batches_csv),
            rows,
            fieldnames=["batch","batch_size","time_s","energy_j","avg_power_w","j_per_sample","throughput_img_s","loss"],
        )
        save_rows_csv(
            os.path.join(self.out_dir, self.export.epochs_csv),
            self.summary.get("epoch_energies", []),
            fieldnames=["epoch","time_s","energy_j","avg_power_w"],
        )

        epoch_pairs = pair_marks(self.rec.marks, "epoch_start", "epoch_end")
        self.summary["epoch_avg_gpu_util_pct"] = _avg_util_for_pairs(self.rec.samples, epoch_pairs, "gpu_util")
        self.summary["epoch_avg_mem_util_pct"] = _avg_util_for_pairs(self.rec.samples, epoch_pairs, "mem_util")
        self.summary["gpu_name"] = self.gpu_name

        with open(os.path.join(self.out_dir, self.export.summary_json), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)


# Convenience factories
def profile_tf(out_dir: Optional[str] = "run_tf", device_index=0, interval_s=0.02, verbose: int = 0) -> PowerDLTFProfiler:
    return PowerDLTFProfiler(out_dir=out_dir, device_index=device_index, interval_s=interval_s, verbose=verbose)

def profile_torch(out_dir: Optional[str] = "run_torch", device_index=0, interval_s=0.02, amp=True, verbose: int = 0) -> PowerDLTorchProfiler:
    return PowerDLTorchProfiler(out_dir=out_dir, device_index=device_index, interval_s=interval_s, amp=amp, verbose=verbose)

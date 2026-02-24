from __future__ import annotations 
from .analysis import integrate_joules, pair_marks


import tensorflow as tf


class PowerCallback(tf.keras.callbacks.Callback):
    def __init__(self, recorder: any):
        super().__init__()
        self.rec = recorder

    def on_train_begin(self, logs=None):
        self.rec.mark("train_start")
        self.rec.start()

    def on_train_end(self, logs=None):
        self.rec.mark("train_end")

    def on_epoch_begin(self, epoch, logs=None):
        self.rec.mark("epoch_start", epoch=int(epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.rec.mark("epoch_end", epoch=int(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        self.rec.mark("batch_start", batch=int(batch))

    def on_train_batch_end(self, batch, logs=None):
        meta = {"batch": int(batch)}
        if logs and "loss" in logs:
            meta["loss"] = float(logs["loss"])
        # Keras bazen size verir (her zaman değil)
        if logs and "size" in logs:
            meta["batch_size"] = int(logs["size"])
        self.rec.mark("batch_end", **meta)

def _avg_power_w(samples, t0, t1):
    e = integrate_joules(samples, t0, t1)
    dt = max(1e-9, (t1 - t0))
    return e / dt

def summarize_energy(recorder, batch_size: int | None = None) -> dict:
    out = {}

    # total train
    total_pairs = pair_marks(recorder.marks, "train_start", "train_end")
    if total_pairs:
        a, b = total_pairs[0]
        E = integrate_joules(recorder.samples, a.t, b.t)
        out["train_energy_j"] = E
        out["train_avg_power_w"] = _avg_power_w(recorder.samples, a.t, b.t)
        out["train_time_s"] = (b.t - a.t)

    # per epoch
    epoch_pairs = pair_marks(recorder.marks, "epoch_start", "epoch_end")
    epoch_energies = []
    for a, b in epoch_pairs:
        epoch = a.meta.get("epoch", None)
        e = integrate_joules(recorder.samples, a.t, b.t)
        epoch_energies.append({
            "epoch": epoch,
            "energy_j": e,
            "time_s": (b.t - a.t),
            "avg_power_w": _avg_power_w(recorder.samples, a.t, b.t),
        })
    out["epoch_energies"] = epoch_energies

    # per batch (step)
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
            "energy_j": e,
            "time_s": dt,
            "avg_power_w": (e / dt) if dt > 0 else None,
            "batch_size": bs,
            "j_per_sample": j_per_sample,
            "throughput_img_s": ips,
            "loss": b.meta.get("loss", None),
        })
    out["batch_rows"] = batch_rows
    return out

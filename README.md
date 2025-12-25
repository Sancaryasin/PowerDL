# PowerDL (SoftwareX package draft)

PowerDL is a lightweight GPU power/energy profiling toolkit for **TensorFlow** and **PyTorch**.
It records GPU power (Watts) via NVML at a fixed sampling interval, marks phase boundaries
(warm-up, training, inference), and provides **in-memory** reports with optional artifact export.

## Quickstart

### TensorFlow

```python
from powerdl.highlevel import profile_tf

with profile_tf(out_dir=None, device_index=0, interval_s=0.02, verbose=1) as prof:
    prof.fit(model, ds, epochs=10, verbose=2)
    prof.infer_keras(model, x_inf, batch_size=512)

rep = prof.report()
rep.plot(figs=["power_time", "cumulative_energy", "gpu_util_time"], out_dir="results/tf", show=False)
# rep.export("runs/tf_mnist")  # optional persistence
```

### PyTorch

```python
from powerdl.highlevel_torch import profile_torch

with profile_torch(out_dir=None, interval_s=0.02, verbose=1) as prof:
    prof.train(model, trainloader, optimizer, loss_fn, epochs=3)
    prof.infer_tensor(model, batch_size=256, n_samples=20000)

rep = prof.report()
rep.plot(all=True, out_dir="results/torch", show=False, smooth=5, shade_phases=True)
# rep.export("runs/torch_cifar")  # optional persistence
```

## Figures (selective + optional)

List available plots:

```python
rep.list_figures()
```

Generate a single figure:

```python
rep.plot_one("power_util_scatter", out_dir="results/figs", show=False)
```

## Notes

* Default mode is **in-memory** (no local files written).
* `export()` enables reproducible artifacts: `summary_*.json`, `samples_*.csv`, `marks_*.csv`, and figures.

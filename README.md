# PowerDL – Unified In-Memory GPU Energy Profiling and Visualization Toolkit

PowerDL is a lightweight, framework-agnostic toolkit for **GPU energy, power, and utilization profiling** of deep learning workloads.  
It supports **PyTorch and TensorFlow**, operates **fully in-memory by default**, and provides **rich, publication-ready visual analytics** with optional artifact export.

The toolkit is designed for:
- Interactive experimentation (no mandatory file I/O)  
- Energy-aware deep learning research  
- Reproducible benchmarking  
- Software-oriented scientific contributions  

---

## Key Features

- Unified high-level API: `profile_torch`, `profile_tf`  
- Fully **in-memory profiling** (no files required by default)  
- Optional export for reproducibility  
- High-resolution **time-series sampling** (NVML-based)  
- Phase-aware analysis (training / inference / warmup / session)  
- Batch-level and epoch-level energy breakdown  
- Optional visualization layer (installed separately)  
- Clean separation between measurement and plotting  
- Research-ready structured outputs (CSV + JSON)  

---

## Installation

### Minimal Core Install (No Visualization)

```bash
pip install powerdl
```

### With Visualization Support

```bash
pip install powerdl[viz]
```

### With PyTorch Examples

```bash
pip install powerdl[examples-torch]
```

### With TensorFlow Examples

```bash
pip install powerdl[examples-tf]
```

---

## Requirements

- Python ≥ 3.9  
- NVIDIA GPU  
- NVIDIA driver with NVML support  
- PyTorch or TensorFlow (only if used)  

Visualization dependencies are optional and not required for core functionality.

---

## Quick Start – PyTorch

```python
from powerdl.highlevel import profile_torch

with profile_torch(
    out_dir=None,
    interval_s=0.02,
    verbose=1
) as prof:

    prof.train(model, trainloader, optimizer, loss_fn, epochs=3)
    prof.infer_tensor(model, batch_size=256, n_samples=20000)

rep = prof.report()
rep.export("results/torch_run")
```

---

## Quick Start – TensorFlow

```python
from powerdl.highlevel import profile_tf

with profile_tf(
    out_dir=None,
    interval_s=0.02,
    verbose=1
) as prof:

    prof.fit(model, dataset, epochs=3)
    prof.infer_keras(model, x_inf, batch_size=512)

rep = prof.report()
rep.export("results/tf_run")
```

---

## In-Memory First Design

PowerDL always collects measurements **in memory first**.

Collected signals:
- GPU power (W)  
- GPU utilization (%)  
- Memory utilization (%)  
- Precise timestamps  
- Phase markers (training, inference, epochs, warmup)  

No files are written unless explicitly requested.

Optional export:

```python
rep.export("runs/experiment_export")
```

---

## Report Object

After profiling:

```python
rep = prof.report()
```

The unified `Report` object contains:
- `samples_df` – full time-series measurements  
- `marks_df` – event and phase markers  
- `summary` – structured scalar metrics  

Visualization is **lazy-loaded** and only requires matplotlib if plotting is requested.

---

## Visualization System (Optional)

Install:

```bash
pip install powerdl[viz]
```

List available figures:

```python
rep.list_figures()
```

---

## Supported Figures

### Time-Series
- `power_time`
- `gpu_util_time`
- `mem_util_time`
- `util_time_dual`
- `energy_rate`
- `power_derivative`
- `cumulative_energy`

### Distribution-Based
- `power_hist`
- `util_hist`
- `power_ecdf`
- `util_ecdf`

### Phase-Aware Statistics
- `power_boxplot_phase`
- `util_boxplot_phase`
- `phase_energy_bar`
- `phase_time_bar`

### Correlation & Density
- `power_util_scatter`
- `power_util_hexbin`

---

## Plotting API

### Plot All Figures

```python
rep.plot(
    all=True,
    out_dir="results/figs",
    show=False,
    smooth=5,
    shade_phases=True
)
```

### Plot Selected Figures

```python
rep.plot(
    figs=[
        "power_time",
        "cumulative_energy",
        "power_util_hexbin",
        "phase_energy_bar"
    ],
    out_dir="results/figs",
    show=True
)
```

### Plot a Single Figure

```python
rep.plot_one("power_ecdf", show=True)
```

Figures are consistently formatted, publication-ready, and phase-aware where applicable.

---

## Exported Artifacts

```python
rep.export("runs/experiment_01")
```

Produces:
- `summary.json`
- `samples.csv`
- `marks.csv`
- `batches.csv` (if batch profiling used)
- `epochs.csv` (if epoch markers exist)

This supports fully reproducible experiments.

---

## Advanced Features

### Repeatability Experiments
- Multi-run stability evaluation  
- Median ± standard deviation reporting  
- Coefficient of variation (CV%) analysis  
- Sampling interval sensitivity testing  

### Memory Safety
- Optional memory warning thresholds  
- Configurable sample buffer limits  
- Auto-flush support for long experiments  

### Framework-Agnostic Architecture
Core measurement layer is independent of:
- PyTorch  
- TensorFlow  
- Matplotlib  

Optional dependencies are imported lazily.

---

## Design Philosophy

PowerDL separates:
- Measurement (NVML sampling)  
- Aggregation (energy integration and phase pairing)  
- Visualization (registry-based plotting)  

It focuses on **temporal and phase-aware energy behavior**, not just scalar totals.

The architecture ensures:
- Minimal overhead  
- Clean packaging  
- Optional visualization  
- Reproducible research design  

---

## Typical Use Cases

- Green AI benchmarking  
- Training vs inference energy comparison  
- GPU saturation analysis  
- Phase-wise energy profiling  
- Deep learning energy efficiency studies  
- Software reproducibility experiments  

---

## License

MIT License

---

## Author

Yasin SANCAR  
Computer Engineer  
Energy-Aware Deep Learning Research

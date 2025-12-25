# PowerDL – Unified In-Memory GPU Energy Profiling and Visualization Toolkit

PowerDL is a lightweight, framework-agnostic toolkit for **GPU energy, power, and utilization profiling** of deep learning workloads.  
It supports **PyTorch and TensorFlow**, operates **fully in-memory by default**, and provides **rich, publication-ready visual analytics** with optional artifact export.

The toolkit is designed for:
- Interactive experimentation (no mandatory file I/O)
- Energy-aware deep learning research
- SoftwareX-style reproducible software contributions

---

## Key Features

- PyTorch and TensorFlow support  
- Fully **in-memory profiling** (no files required by default)  
- Optional export for reproducibility  
- High-resolution **time-series sampling** (NVML-based)  
- Rich, selectable **visual analytics**  
- Uniform API across frameworks  
- SoftwareX-ready outputs (figures + summaries)  

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-repo/powerdl.git
cd powerdl
pip install -e .
```

### Requirements
- Python ≥ 3.9  
- NVIDIA GPU with NVML support  
- PyTorch or TensorFlow  
- numpy, pandas, matplotlib  

---

## Quick Start

### PyTorch Example

```python
from powerdl.highlevel_torch import profile_torch

with profile_torch(out_dir=None, interval_s=0.02, verbose=1) as prof:
    prof.train(model, trainloader, optimizer, loss_fn, epochs=3)
    prof.infer_tensor(model, batch_size=256, n_samples=20000)

rep = prof.report()
rep.plot(all=True, out_dir="results/torch_figs")
```

### TensorFlow Example

```python
from powerdl.highlevel import profile_tf

with profile_tf(out_dir=None, device_index=0, interval_s=0.02, verbose=1) as prof:
    prof.fit(model, dataset, epochs=10)
    prof.infer_keras(model, x_inf, batch_size=512)

rep = prof.report()
rep.plot(all=True, out_dir="results/tf_figs")
```

---

## In-Memory First Design

PowerDL always collects measurements **in memory**:

- GPU power (W)  
- GPU utilization (%)  
- Memory utilization (%)  
- Time stamps  
- Phase markers (training / inference / epochs)  

No files are written unless explicitly requested.

Optional export:

```python
rep.export("runs/experiment_export")
```

---

## Report Object

After profiling, a unified `Report` object is returned:

```python
rep = prof.report()
```

The report contains:
- `samples` – time-series measurements  
- `marks` – phase and event markers  
- `summary` – aggregated scalar metrics  

---

## Visualization System

PowerDL includes a **registry-based visualization system**.  
Users can list, select, or generate all available figures.

### List Available Figures

```python
rep.list_figures()
```

---

## Supported Figures (Complete List)

### Time-Series
- `power_time` – GPU power vs time (phase-shaded)  
- `gpu_util_time` – GPU utilization vs time  
- `mem_util_time` – memory utilization vs time  
- `rolling_power` – smoothed power trace  
- `rolling_util` – smoothed utilization trace  

### Energy-Oriented
- `cumulative_energy` – cumulative energy over time  
- `energy_rate` – instantaneous energy rate  
- `phase_energy_bar` – training vs inference energy breakdown  
- `epoch_energy` – energy per epoch (if epoch markers exist)  

### Distribution-Based
- `power_hist` – power histogram  
- `util_hist` – utilization histogram  
- `power_ecdf` – empirical CDF of power  
- `util_ecdf` – empirical CDF of utilization  

### Phase-Aware Statistics
- `power_boxplot_phase` – power by phase  
- `util_boxplot_phase` – utilization by phase  
- `phase_time_bar` – time spent per phase  

### Correlation & Density
- `power_util_scatter` – power vs GPU utilization  
- `power_util_hexbin` – density-based power–utilization relation  

### Throughput & Efficiency
- `throughput_window` – instantaneous throughput  
- `efficiency_curve` – energy–performance trade-off  

---

## Plotting API

### Plot All Figures

```python
rep.plot(all=True, out_dir="results/figs", show=False)
```

### Plot Selected Figures Only

```python
rep.plot(
    figs=[
        "power_time",
        "cumulative_energy",
        "power_util_hexbin",
        "phase_energy_bar"
    ],
    out_dir="results/figs",
    show=True,
    smooth=5,
    shade_phases=True
)
```

### Plot a Single Figure

```python
rep.plot_one("power_ecdf", show=True)
```

---

## Visual Quality

All figures are:
- Consistently sized  
- Grid-aligned with clean axes  
- Top/right spines removed  
- Publication-ready  
- Saved as **PNG and PDF**  

---

## Exported Artifacts (Optional)

```python
rep.export("runs/experiment_01")
```

Produces:
- `summary.json`  
- `samples.csv`  
- `marks.csv`  
- `figures/*.png`  
- `figures/*.pdf`  

This enables full reproducibility without enforcing disk I/O.

---

## Design Rationale (SoftwareX)

PowerDL separates:
- Measurement  
- Analysis  
- Visualization  

It focuses on **temporal and phase-aware energy behavior**, rather than only scalar energy metrics.  
The toolkit supports both interactive workflows and reproducible research.

---

## Typical Use Cases

- Energy-aware deep learning research  
- Green AI benchmarking  
- Training vs inference energy comparison  
- GPU efficiency and saturation analysis  
- Software reproducibility studies  

---

## License

MIT License

---

## Citation

If you use PowerDL in your work, please cite:

> PowerDL: An In-Memory GPU Energy Profiling and Visualization Toolkit for Deep Learning Frameworks

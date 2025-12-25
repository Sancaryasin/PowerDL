from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PlotSpec:
    key: str
    title: str
    fn: Callable[..., Any]
    description: str


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _phase_intervals(marks_df) -> Dict[str, Tuple[float, float]]:
    """Return best-effort phase intervals from marks (train/infer/warmup/session)."""
    intervals = {}
    for phase in ["train", "infer", "warmup", "session"]:
        s = marks_df[marks_df["name"] == f"{phase}_start"]
        e = marks_df[marks_df["name"] == f"{phase}_end"]
        if len(s) and len(e):
            intervals[phase] = (float(s.iloc[0]["t"]), float(e.iloc[-1]["t"]))
    return intervals


def _slice_by_interval(df, t_col: str, start_t: float, end_t: float):
    m = (df[t_col].astype(float) >= float(start_t)) & (df[t_col].astype(float) <= float(end_t))
    return df.loc[m]


def _add_phase_shading(ax, *, df, marks_df):
    if marks_df is None or marks_df.empty or df.empty:
        return
    intervals = _phase_intervals(marks_df)
    if not intervals:
        return
    t0 = float(df["t"].min())
    # Shade + annotate lightly (no legend clutter)
    for phase, (a, b) in intervals.items():
        ax.axvspan(a - t0, b - t0, alpha=0.08)
        mid = (a + b) / 2.0
        ax.text(mid - t0, ax.get_ylim()[1], phase, va="top", ha="center", fontsize=9, alpha=0.6)


def plot_power_time(rep, ax, *, smooth: int = 0, shade_phases: bool = True, show_raw: bool = True, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    y = df["p_w"].astype(float)
    x = df["ts"].astype(float)
    if show_raw:
        ax.plot(x, y, alpha=0.35, linewidth=1.0)
    if smooth and smooth > 1:
        ys = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, ys, linewidth=2.0)
    else:
        ax.plot(x, y, linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")

    if shade_phases:
        _add_phase_shading(ax, df=df, marks_df=rep.marks_df)
    return True


def plot_cumulative_energy(rep, ax, **_):
    import numpy as np

    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    t = df["t"].to_numpy(dtype=float)
    p = df["p_w"].to_numpy(dtype=float)
    # cumulative trapezoidal integration
    dt = np.diff(t, prepend=t[0])
    e = np.cumsum(p * dt)
    ax.plot(df["ts"], e)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    return True


def plot_power_derivative(rep, ax, *, smooth: int = 0, **_):
    """dP/dt approximated to reveal bursts/throttling."""
    import numpy as np

    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    t = df["t"].to_numpy(dtype=float)
    p = df["p_w"].astype(float)
    if smooth and smooth > 1:
        p = p.rolling(smooth, min_periods=1).mean()
    p = p.to_numpy(dtype=float)
    dt = np.diff(t)
    dp = np.diff(p)
    if len(dt) == 0:
        return False
    d = dp / np.where(dt == 0, 1e-9, dt)
    ax.plot(df["ts"].iloc[1:].astype(float), d, linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dPower/dt (W/s)")
    return True


def plot_energy_rate(rep, ax, *, smooth: int = 0, **_):
    """Instantaneous energy rate (J/s) == power, but we plot with rolling stats & quantiles."""
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    p = df["p_w"].astype(float)
    if smooth and smooth > 1:
        p_s = p.rolling(smooth, min_periods=1).mean()
        ax.plot(df["ts"].astype(float), p_s, linewidth=2.0)
    else:
        ax.plot(df["ts"].astype(float), p, linewidth=2.0)
    # Quantile band (robust)
    try:
        q10 = float(p.quantile(0.10))
        q90 = float(p.quantile(0.90))
        ax.axhspan(q10, q90, alpha=0.08)
    except Exception:
        pass
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy rate (J/s)")
    return True


def plot_gpu_util_time(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False
    y = df["gpu_util"].astype(float)
    if y.isna().all():
        return False
    if smooth and smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()
    ax.plot(df["ts"], y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU util (%)")
    ax.set_ylim(0, 100)
    return True


def plot_mem_util_time(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "mem_util" not in df.columns:
        return False
    y = df["mem_util"].astype(float)
    if y.isna().all():
        return False
    if smooth and smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()
    ax.plot(df["ts"], y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mem util (%)")
    ax.set_ylim(0, 100)
    return True


def plot_util_time_dual(rep, ax, *, smooth: int = 0, **_):
    """GPU util + mem util on one figure (if both exist)."""
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False
    g = df["gpu_util"].astype(float)
    if g.isna().all():
        return False
    if smooth and smooth > 1:
        g = g.rolling(smooth, min_periods=1).mean()
    ax.plot(df["ts"].astype(float), g, linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU util (%)")
    ax.set_ylim(0, 100)
    if "mem_util" in df.columns:
        m = df["mem_util"].astype(float)
        if not m.isna().all():
            if smooth and smooth > 1:
                m = m.rolling(smooth, min_periods=1).mean()
            ax2 = ax.twinx()
            ax2.plot(df["ts"].astype(float), m, linewidth=1.8, alpha=0.7)
            ax2.set_ylabel("Mem util (%)")
            ax2.set_ylim(0, 100)
    return True


def plot_power_hist(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    ax.hist(df["p_w"].astype(float).dropna().values, bins=40)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Count")
    return True


def plot_util_hist(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False
    s = df["gpu_util"].dropna()
    if s.empty:
        return False
    ax.hist(s.astype(float).values, bins=40)
    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("Count")
    return True


def plot_power_ecdf(rep, ax, **_):
    import numpy as np
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False
    x = df["p_w"].astype(float).dropna().to_numpy()
    if x.size == 0:
        return False
    x = np.sort(x)
    y = np.linspace(0, 1, x.size, endpoint=True)
    ax.plot(x, y, linewidth=2.0)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1)
    return True


def plot_util_ecdf(rep, ax, **_):
    import numpy as np
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False
    x = df["gpu_util"].astype(float).dropna().to_numpy()
    if x.size == 0:
        return False
    x = np.sort(x)
    y = np.linspace(0, 1, x.size, endpoint=True)
    ax.plot(x, y, linewidth=2.0)
    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1)
    return True


def plot_power_util_scatter(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns or "p_w" not in df.columns:
        return False
    s = df[["gpu_util", "p_w"]].dropna()
    if s.empty:
        return False
    ax.scatter(s["gpu_util"].astype(float), s["p_w"].astype(float), s=8, alpha=0.6)
    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("Power (W)")
    return True


def plot_power_util_hexbin(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns or "p_w" not in df.columns:
        return False
    s = df[["gpu_util", "p_w"]].dropna()
    if s.empty:
        return False
    hb = ax.hexbin(s["gpu_util"].astype(float), s["p_w"].astype(float), gridsize=30, mincnt=1)
    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("Power (W)")
    try:
        import matplotlib.pyplot as plt
        plt.colorbar(hb, ax=ax, label="count")
    except Exception:
        pass
    return True


def plot_power_boxplot_by_phase(rep, ax, **_):
    df = rep.samples_df
    m = rep.marks_df
    if df.empty or m is None or m.empty or "p_w" not in df.columns:
        return False
    intervals = _phase_intervals(m)
    if not intervals:
        return False
    data, labels = [], []
    for phase in ["warmup", "train", "infer", "session"]:
        if phase in intervals:
            a, b = intervals[phase]
            d = _slice_by_interval(df, "t", a, b)["p_w"].astype(float).dropna().values
            if len(d):
                data.append(d)
                labels.append(phase)
    if not data:
        return False
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Power (W)")
    return True


def plot_util_boxplot_by_phase(rep, ax, **_):
    df = rep.samples_df
    m = rep.marks_df
    if df.empty or m is None or m.empty or "gpu_util" not in df.columns:
        return False
    intervals = _phase_intervals(m)
    if not intervals:
        return False
    data, labels = [], []
    for phase in ["warmup", "train", "infer", "session"]:
        if phase in intervals:
            a, b = intervals[phase]
            d = _slice_by_interval(df, "t", a, b)["gpu_util"].astype(float).dropna().values
            if len(d):
                data.append(d)
                labels.append(phase)
    if not data:
        return False
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("GPU util (%)")
    ax.set_ylim(0, 100)
    return True


def plot_phase_energy_bar(rep, ax, **_):
    # Requires summary fields from summarize_energy/summarize_session
    s = rep.summary or {}
    train_e = _safe_float(s.get("train_energy_j"))
    infer_e = _safe_float(s.get("infer_energy_j"))
    if train_e is None and infer_e is None:
        return False
    labels, vals = [], []
    if train_e is not None:
        labels.append("train")
        vals.append(train_e)
    if infer_e is not None:
        labels.append("infer")
        vals.append(infer_e)
    ax.bar(labels, vals)
    ax.set_ylabel("Energy (J)")
    return True


def plot_phase_time_bar(rep, ax, **_):
    s = rep.summary or {}
    train_t = _safe_float(s.get("train_time_s"))
    infer_t = _safe_float(s.get("infer_time_s"))
    if train_t is None and infer_t is None:
        return False
    labels, vals = [], []
    if train_t is not None:
        labels.append("train")
        vals.append(train_t)
    if infer_t is not None:
        labels.append("infer")
        vals.append(infer_t)
    ax.bar(labels, vals)
    ax.set_ylabel("Time (s)")
    return True


# Registry
PLOT_REGISTRY: Dict[str, PlotSpec] = {
    "power_time": PlotSpec(
        key="power_time",
        title="GPU Power over Time",
        fn=plot_power_time,
        description="Time-series GPU power trace; optionally shaded phases (train/infer/warmup).",
    ),
    "cumulative_energy": PlotSpec(
        key="cumulative_energy",
        title="Cumulative Energy",
        fn=plot_cumulative_energy,
        description="Cumulative energy curve computed by integrating power over time.",
    ),
    "energy_rate": PlotSpec(
        key="energy_rate",
        title="Energy Rate (Power) with Quantile Band",
        fn=plot_energy_rate,
        description="Power time-series with optional smoothing and 10–90% band.",
    ),
    "power_derivative": PlotSpec(
        key="power_derivative",
        title="Power Change Rate (dP/dt)",
        fn=plot_power_derivative,
        description="Approximate dPower/dt to highlight bursts/throttling.",
    ),
    "gpu_util_time": PlotSpec(
        key="gpu_util_time",
        title="GPU Utilization over Time",
        fn=plot_gpu_util_time,
        description="GPU utilization (%) time-series.",
    ),
    "mem_util_time": PlotSpec(
        key="mem_util_time",
        title="Memory Utilization over Time",
        fn=plot_mem_util_time,
        description="Memory utilization (%) time-series (if available).",
    ),
    "util_time_dual": PlotSpec(
        key="util_time_dual",
        title="GPU + Memory Utilization over Time",
        fn=plot_util_time_dual,
        description="Dual-axis utilization time-series (GPU + memory) if available.",
    ),
    "power_hist": PlotSpec(
        key="power_hist",
        title="Power Distribution",
        fn=plot_power_hist,
        description="Histogram of sampled GPU power.",
    ),
    "util_hist": PlotSpec(
        key="util_hist",
        title="GPU Utilization Distribution",
        fn=plot_util_hist,
        description="Histogram of sampled GPU utilization.",
    ),
    "power_ecdf": PlotSpec(
        key="power_ecdf",
        title="Power ECDF",
        fn=plot_power_ecdf,
        description="Empirical CDF of sampled power values.",
    ),
    "util_ecdf": PlotSpec(
        key="util_ecdf",
        title="GPU Utilization ECDF",
        fn=plot_util_ecdf,
        description="Empirical CDF of sampled GPU utilization values.",
    ),
    "power_util_scatter": PlotSpec(
        key="power_util_scatter",
        title="Power vs GPU Utilization",
        fn=plot_power_util_scatter,
        description="Scatter plot of power vs GPU utilization.",
    ),
    "power_util_hexbin": PlotSpec(
        key="power_util_hexbin",
        title="Power vs GPU Utilization (Hexbin)",
        fn=plot_power_util_hexbin,
        description="Density view (hexbin) of power vs GPU utilization.",
    ),
    "power_boxplot_phase": PlotSpec(
        key="power_boxplot_phase",
        title="Power by Phase (Boxplot)",
        fn=plot_power_boxplot_by_phase,
        description="Phase-wise distribution of power (warmup/train/infer/session).",
    ),
    "util_boxplot_phase": PlotSpec(
        key="util_boxplot_phase",
        title="GPU Utilization by Phase (Boxplot)",
        fn=plot_util_boxplot_by_phase,
        description="Phase-wise distribution of GPU utilization.",
    ),
    "phase_energy_bar": PlotSpec(
        key="phase_energy_bar",
        title="Energy Breakdown (Train vs Infer)",
        fn=plot_phase_energy_bar,
        description="Bar chart of energy for training and inference phases from summary.",
    ),
    "phase_time_bar": PlotSpec(
        key="phase_time_bar",
        title="Time Breakdown (Train vs Infer)",
        fn=plot_phase_time_bar,
        description="Bar chart of time for training and inference phases from summary.",
    ),
}


def list_figures() -> List[str]:
    return sorted(PLOT_REGISTRY.keys())

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# --- Optional plotting deps (installed via: pip install powerdl[viz]) ---
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (prevents GUI blocking)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
except ImportError as e:
    raise ImportError(
        "Plotting requires optional dependencies. "
        "Install with: pip install powerdl[viz]"
    ) from e

# SciencePlots (base look) - optional
try:
    import scienceplots   
    plt.style.use(["science"])
except Exception:
    # Fallback: default Matplotlib style
    pass


def _set_percent_xlim(ax, x, *, pad=0.08, min_max=25, cap=100):
    """
    x: expected percent values (0-100)
    - always show 0
    - expand right bound based on data (avoid too much empty space)
    - but do not force 100
    """
    try:
        xmax = float(np.nanmax(x))
    except Exception:
        ax.set_xlim(0, cap)
        return

    if not np.isfinite(xmax) or xmax <= 0:
        ax.set_xlim(0, cap)
        return

    right = xmax * (1.0 + pad)
    right = max(right, float(min_max))
    right = min(right, float(cap))
    ax.set_xlim(0, right)


# ---- Global "pro" visual defaults (colorful + clean) ----
PRO_COLORS = {
    "train": "#d62728",   # red
    "infer": "#1f77b4",   # blue
    "warmup": "#ff7f0e",  # orange
    "session": "#2ca02c", # green
    "neutral": "#2b2b2b",
    "grid": "#d8d8d8",
    "bg_fig": "#f6f7fb",
    "bg_ax": "#ffffff",
    "band": "#1f77b4",
    "mean": "#d62728",
}

plt.rcParams.update({
    # background
    "figure.facecolor": PRO_COLORS["bg_fig"],
    "axes.facecolor": PRO_COLORS["bg_ax"],

    # text
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # axes
    "axes.edgecolor": "#3a3a3a",
    "axes.linewidth": 1.0,

    # grid
    "axes.grid": True,
    "grid.color": PRO_COLORS["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,

    # lines
    "lines.linewidth": 2.6,
    "lines.solid_capstyle": "round",

    # legend
    "legend.frameon": False,

    # save defaults (report.py controls actual save)
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
])


# -------- labels (units) --------
def _xlabel_time(ax):
    ax.set_xlabel("Time (s)")


def _ylabel_power(ax):
    ax.set_ylabel("Power (W)")


def _ylabel_energy(ax):
    ax.set_ylabel("Energy (J)")


def _ylabel_power_rate(ax):
    # energy rate == power
    ax.set_ylabel("Power (W)")


def _ylabel_util(ax, what="GPU"):
    ax.set_ylabel(f"{what} util (%)")


@dataclass(frozen=True)
class PlotSpec:
    key: str
    title: str
    fn: Callable[..., Any]
    description: str


def _fmt_thousands(x, _pos=None):
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _phase_intervals(marks_df) -> Dict[str, Tuple[float, float]]:
    """Return best-effort phase intervals from marks (train/infer/warmup/session)."""
    intervals: Dict[str, Tuple[float, float]] = {}
    for phase in ["warmup", "train", "infer", "session"]:
        s = marks_df[marks_df["name"] == f"{phase}_start"]
        e = marks_df[marks_df["name"] == f"{phase}_end"]
        if len(s) and len(e):
            intervals[phase] = (float(s.iloc[0]["t"]), float(e.iloc[-1]["t"]))
    return intervals


def _slice_by_interval(df, t_col: str, start_t: float, end_t: float):
    m = (df[t_col].astype(float) >= float(start_t)) & (df[t_col].astype(float) <= float(end_t))
    return df.loc[m]


def _polish_axes(ax):
    """Consistent, clean axes."""
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9, colors="#333333")


def _add_phase_shading(ax, *, df, marks_df):
    if marks_df is None or getattr(marks_df, "empty", True) or df.empty:
        return
    intervals = _phase_intervals(marks_df)
    if not intervals:
        return

    t0 = float(df["t"].min())

    # freeze y-limits BEFORE annotation to avoid shifting
    y0, y1 = ax.get_ylim()
    y_text = y1 - (y1 - y0) * 0.02  # slightly below top

    for phase, (a, b) in intervals.items():
        color = PRO_COLORS.get(phase, "#999999")
        ax.axvspan(a - t0, b - t0, alpha=0.08, color=color, lw=0)
        mid = (a + b) / 2.0 - t0
        ax.text(mid, y_text, phase, va="top", ha="center", fontsize=9, color="#333333", alpha=0.75)

    ax.set_ylim(y0, y1)


# -------------------- PLOTS --------------------

def plot_power_time(rep, ax, *, smooth: int = 0, shade_phases: bool = True, show_raw: bool = True, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    y = df["p_w"].astype(float)
    x = df["ts"].astype(float)

    if show_raw:
        ax.plot(x, y, alpha=0.25, linewidth=1.2, color="#7f7f7f", label="raw")

    if smooth and smooth > 1:
        ys = y.rolling(smooth, min_periods=1).mean()
        ax.plot(x, ys, linewidth=2.8, color=PRO_COLORS["infer"], label=f"smoothed (w={smooth})")
        series_for_stats = ys
    else:
        ax.plot(x, y, linewidth=2.8, color=PRO_COLORS["infer"], label="power")
        series_for_stats = y

    try:
        m = float(series_for_stats.mean())
        ax.axhline(m, linestyle="--", linewidth=2.0, color=PRO_COLORS["mean"], alpha=0.9)
        ax.text(
            0.99, m, f" mean={m:.1f} W",
            ha="right", va="bottom",
            transform=ax.get_yaxis_transform(),
            color=PRO_COLORS["mean"], fontsize=10
        )
    except Exception:
        pass

    _xlabel_time(ax)
    _ylabel_power(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    if shade_phases:
        _add_phase_shading(ax, df=df, marks_df=rep.marks_df)

    return True


def plot_cumulative_energy(rep, ax, shade_phases: bool = True, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    t = df["t"].to_numpy(dtype=float)
    p = df["p_w"].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])
    e = np.cumsum(p * dt)

    ax.plot(df["ts"].astype(float), e, color=PRO_COLORS["session"], linewidth=3.0)
    _xlabel_time(ax)
    _ylabel_energy(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    try:
        ax.scatter(df["ts"].iloc[-1], e[-1], s=35, color=PRO_COLORS["train"], zorder=5)
        ax.text(
            df["ts"].iloc[-1], e[-1], f"  {e[-1]:,.1f} J",
            va="center", ha="left", fontsize=10, color="#222222"
        )
    except Exception:
        pass

    if shade_phases:
        _add_phase_shading(ax, df=df, marks_df=rep.marks_df)

    return True


def plot_power_derivative(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

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

    ax.plot(df["ts"].iloc[1:].astype(float), d, linewidth=2.4, color="#9467bd")
    ax.axhline(0.0, linewidth=1.5, color="#444444", alpha=0.6)
    _xlabel_time(ax)
    ax.set_ylabel("dPower/dt (W/s)")
    return True


def plot_energy_rate(rep, ax, *, smooth: int = 0, shade_phases: bool = True, **_):
    # NOTE: This plot is effectively power vs time (energy rate == power).
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    p = df["p_w"].astype(float)
    x = df["ts"].astype(float)

    if smooth and smooth > 1:
        y = p.rolling(smooth, min_periods=1).mean()
        ax.plot(x, y, linewidth=3.0, color=PRO_COLORS["infer"], label="smoothed power")
        base = y
    else:
        ax.plot(x, p, linewidth=3.0, color=PRO_COLORS["infer"], label="power")
        base = p

    try:
        q10 = float(p.quantile(0.10))
        q90 = float(p.quantile(0.90))
        ax.axhspan(q10, q90, alpha=0.10, color=PRO_COLORS["band"], lw=0)
        ax.text(
            0.01, 0.97, "10–90% band",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="#333333"
        )
    except Exception:
        pass

    try:
        m = float(base.mean())
        ax.axhline(m, linestyle="--", linewidth=2.0, color=PRO_COLORS["mean"], alpha=0.9)
        ax.text(
            0.99, m, f" mean={m:.1f} W",
            ha="right", va="bottom",
            transform=ax.get_yaxis_transform(),
            color=PRO_COLORS["mean"], fontsize=10
        )
    except Exception:
        pass

    _xlabel_time(ax)
    _ylabel_power_rate(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    if shade_phases:
        _add_phase_shading(ax, df=df, marks_df=rep.marks_df)

    return True


def plot_gpu_util_time(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False
    y = df["gpu_util"].astype(float)
    if y.isna().all():
        return False

    _polish_axes(ax)

    if smooth and smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()

    ax.plot(df["ts"].astype(float), y, color=PRO_COLORS["warmup"], linewidth=2.8)
    _xlabel_time(ax)
    _ylabel_util(ax, "GPU")
    ax.set_ylim(0, 100)
    return True


def plot_mem_util_time(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "mem_util" not in df.columns:
        return False
    y = df["mem_util"].astype(float)
    if y.isna().all():
        return False

    _polish_axes(ax)

    if smooth and smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()

    ax.plot(df["ts"].astype(float), y, color="#2ca02c", linewidth=2.8)
    _xlabel_time(ax)
    _ylabel_util(ax, "Mem")
    ax.set_ylim(0, 100)
    return True


def plot_util_time_dual(rep, ax, *, smooth: int = 0, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False

    g = df["gpu_util"].astype(float)
    if g.isna().all():
        return False

    _polish_axes(ax)

    if smooth and smooth > 1:
        g = g.rolling(smooth, min_periods=1).mean()

    ax.plot(df["ts"].astype(float), g, linewidth=2.8, color=PRO_COLORS["warmup"], label="GPU util")
    _xlabel_time(ax)
    _ylabel_util(ax, "GPU")
    ax.set_ylim(0, 100)

    if "mem_util" in df.columns:
        m = df["mem_util"].astype(float)
        if not m.isna().all():
            if smooth and smooth > 1:
                m = m.rolling(smooth, min_periods=1).mean()
            ax2 = ax.twinx()
            ax2.plot(df["ts"].astype(float), m, linewidth=2.4, color="#2ca02c", alpha=0.95, label="Mem util")
            ax2.set_ylabel("Mem util (%)")
            ax2.set_ylim(0, 100)
            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)

    return True


def plot_power_hist(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    x = df["p_w"].astype(float).dropna().values
    if x.size == 0:
        return False

    ax.hist(x, bins=40, alpha=0.90, edgecolor="#222222", linewidth=0.6, color=PRO_COLORS["infer"])
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Count")

    try:
        med = float(np.median(x))
        ax.axvline(med, linestyle="--", linewidth=2.0, color=PRO_COLORS["mean"], alpha=0.9)
        ax.text(
            med, ax.get_ylim()[1], f" median={med:.1f} W",
            ha="left", va="top", fontsize=10, color=PRO_COLORS["mean"]
        )
    except Exception:
        pass

    return True


def plot_util_hist(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False

    _polish_axes(ax)

    x = df["gpu_util"].dropna().astype(float).values
    if x.size == 0:
        return False

    ax.hist(x, bins=30, alpha=0.90, edgecolor="#222222", linewidth=0.6, color=PRO_COLORS["warmup"])
    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("Count")
    _set_percent_xlim(ax, x, pad=0.10)
    return True


def plot_power_ecdf(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    x = df["p_w"].astype(float).dropna().to_numpy()
    if x.size == 0:
        return False

    x = np.sort(x)
    y = np.linspace(0, 1, x.size, endpoint=True)

    ax.plot(x, y, linewidth=3.0, color=PRO_COLORS["infer"])
    ax.fill_between(x, 0, y, alpha=0.06, color=PRO_COLORS["infer"])

    for q, lab in [(0.5, "median"), (0.9, "p90")]:
        try:
            xv = float(np.quantile(x, q))
            ax.axvline(xv, linestyle="--", linewidth=2.0, color="#444444", alpha=0.7)
            ax.text(
                xv, 0.02, f" {lab}={xv:.1f}",
                rotation=90, va="bottom", ha="left", fontsize=9, color="#333333"
            )
        except Exception:
            pass

    ax.set_xlabel("Power (W)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1)
    return True


def plot_util_ecdf(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns:
        return False

    _polish_axes(ax)

    x = df["gpu_util"].astype(float).dropna().to_numpy()
    if x.size == 0:
        return False

    x = np.sort(x)
    y = np.linspace(0, 1, x.size, endpoint=True)

    ax.plot(x, y, linewidth=3.0, color=PRO_COLORS["warmup"])
    ax.fill_between(x, 0, y, alpha=0.06, color=PRO_COLORS["warmup"])

    ax.set_xlabel("GPU util (%)")
    ax.set_ylabel("ECDF")
    _set_percent_xlim(ax, x, pad=0.10)
    ax.set_ylim(0, 1)
    return True


def plot_power_util_scatter(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    s = df[["gpu_util", "p_w"]].dropna()
    if s.empty:
        return False

    x = s["gpu_util"].astype(float).to_numpy()
    y = s["p_w"].astype(float).to_numpy()

    ax.scatter(x, y, s=18, alpha=0.45, edgecolors="none", color="#9467bd")
    ax.set_xlabel("GPU util (%)")
    _ylabel_power(ax)
    _set_percent_xlim(ax, x, pad=0.10)
    return True


def plot_power_util_hexbin(rep, ax, **_):
    df = rep.samples_df
    if df.empty or "gpu_util" not in df.columns or "p_w" not in df.columns:
        return False

    _polish_axes(ax)

    s = df[["gpu_util", "p_w"]].dropna()
    if s.empty:
        return False

    x = s["gpu_util"].astype(float).to_numpy()
    y = s["p_w"].astype(float).to_numpy()

    hb = ax.hexbin(x, y, gridsize=32, mincnt=1, linewidths=0.0, alpha=0.95)
    ax.set_xlabel("GPU util (%)")
    _ylabel_power(ax)
    _set_percent_xlim(ax, x, pad=0.10)
    try:
        cb = plt.colorbar(hb, ax=ax, label="Count")
        cb.outline.set_visible(False)
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

    _polish_axes(ax)

    data, labels, colors = [], [], []
    for phase in ["warmup", "train", "infer", "session"]:
        if phase in intervals:
            a, b = intervals[phase]
            d = _slice_by_interval(df, "t", a, b)["p_w"].astype(float).dropna().values
            if len(d):
                data.append(d)
                labels.append(phase)
                colors.append(PRO_COLORS.get(phase, "#999999"))

    if not data:
        return False

    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.0)

    for k in ("medians", "whiskers", "caps"):
        for line in bp.get(k, []):
            line.set_color("#222222")
            line.set_linewidth(1.6)

    _ylabel_power(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    return True


def plot_util_boxplot_by_phase(rep, ax, **_):
    df = rep.samples_df
    m = rep.marks_df
    if df.empty or m is None or m.empty or "gpu_util" not in df.columns:
        return False

    intervals = _phase_intervals(m)
    if not intervals:
        return False

    _polish_axes(ax)

    data, labels, colors = [], [], []
    for phase in ["warmup", "train", "infer", "session"]:
        if phase in intervals:
            a, b = intervals[phase]
            d = _slice_by_interval(df, "t", a, b)["gpu_util"].astype(float).dropna().values
            if len(d):
                data.append(d)
                labels.append(phase)
                colors.append(PRO_COLORS.get(phase, "#999999"))

    if not data:
        return False

    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.0)

    for k in ("medians", "whiskers", "caps"):
        for line in bp.get(k, []):
            line.set_color("#222222")
            line.set_linewidth(1.6)

    _ylabel_util(ax, "GPU")
    ax.set_ylim(0, 100)
    return True


def plot_phase_energy_bar(rep, ax, **_):
    s = rep.summary or {}
    train_e = _safe_float(s.get("train_energy_j"))
    infer_e = _safe_float(s.get("infer_energy_j"))

    if train_e is None and infer_e is None:
        return False

    _polish_axes(ax)

    labels = ["Training", "Inference"]
    values = [train_e if train_e is not None else 0.0,
              infer_e if infer_e is not None else 0.0]

    total = float(np.sum(values)) if np.sum(values) > 0 else 0.0
    vmax = max(values) if values else 1.0
    ax.set_ylim(0, vmax * 1.22 if vmax > 0 else 1.0)

    colors = [PRO_COLORS["train"], PRO_COLORS["infer"]]
    bars = ax.bar(
        labels, values,
        color=colors, width=0.55,
        edgecolor="#222222", linewidth=0.9,
        alpha=0.92, zorder=3
    )

    _ylabel_energy(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    for bar, v in zip(bars, values):
        if v <= 0:
            txt = "n/a"
        else:
            pct = (v / total * 100.0) if total > 0 else 0.0
            txt = f"{v:,.1f} J\n({pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vmax * 0.035,
            txt,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#111111"
        )

    if train_e is not None and infer_e is not None and infer_e > 0:
        ratio = train_e / infer_e
        subtitle = f"Total: {total:,.1f} J   |   Train/Infer: {ratio:.2f}×"
    else:
        subtitle = f"Total: {total:,.1f} J"

    ax.text(
        0.5, 0.97, subtitle,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10, color="#333333"
    )
    return True


def plot_phase_time_bar(rep, ax, **_):
    s = rep.summary or {}
    train_t = _safe_float(s.get("train_time_s"))
    infer_t = _safe_float(s.get("infer_time_s"))

    if train_t is None and infer_t is None:
        return False

    _polish_axes(ax)

    labels = ["Training", "Inference"]
    values = [train_t if train_t is not None else 0.0,
              infer_t if infer_t is not None else 0.0]

    total = float(np.sum(values)) if np.sum(values) > 0 else 0.0
    vmax = max(values) if values else 1.0
    ax.set_ylim(0, vmax * 1.22 if vmax > 0 else 1.0)

    colors = [PRO_COLORS["train"], PRO_COLORS["infer"]]
    bars = ax.bar(
        labels, values,
        color=colors, width=0.55,
        edgecolor="#222222", linewidth=0.9,
        alpha=0.92, zorder=3
    )

    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    for bar, v in zip(bars, values):
        txt = "n/a" if v <= 0 else f"{v:.2f} s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vmax * 0.035,
            txt,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#111111"
        )

    ax.text(
        0.5, 0.97, f"Total: {total:.2f} s",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10, color="#333333"
    )
    return True


# -------------------- Registry --------------------

PLOT_REGISTRY: Dict[str, PlotSpec] = {
    "power_time": PlotSpec(
        key="power_time",
        title="GPU Power over Time",
        fn=plot_power_time,
        description="Time-series GPU power trace; optional smoothing and phase shading.",
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
        description="Power time-series with optional smoothing, quantile band and mean line.",
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
        description="Histogram of sampled GPU power with median marker.",
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
        description="Empirical CDF of sampled power values with percentile markers.",
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
        description="Annotated energy comparison for training vs inference from summary.",
    ),
    "phase_time_bar": PlotSpec(
        key="phase_time_bar",
        title="Time Breakdown (Train vs Infer)",
        fn=plot_phase_time_bar,
        description="Annotated time comparison for training vs inference from summary.",
    ),
}


def list_figures() -> List[str]:
    return sorted(PLOT_REGISTRY.keys())
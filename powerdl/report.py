from __future__ import annotations

import os
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Headless backend to avoid any GUI blocking/hanging
import matplotlib
matplotlib.use("Agg")


def _apply_publication_style(style: str = "science"):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "SciencePlots is required. Install with: pip install SciencePlots"
        ) from e

    # Base Science style
    plt.style.use(["science"])

    # --- MODERN / COLORFUL OVERRIDES ---
    mpl.rcParams.update({
        # Figure & background
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.facecolor": "#f7f7f7",
        "axes.facecolor": "#ffffff",

        # Axes
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",

        # Grid
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,

        # Fonts
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Ticks
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Lines
        "lines.linewidth": 2.5,
        "lines.markersize": 6,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,

        # Save
        "savefig.bbox": "tight",
    })

    # 🎨 MODERN COLOR CYCLE (tableau + plotly vibe)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
        ]
    )


@dataclass
class Report:
    """In-memory report produced by PowerDL profilers.

    The report is designed to work without writing intermediate CSV files.
    Users may optionally export artifacts (summary + samples + marks + figures).
    """

    backend: str
    samples: List[Any]
    marks: List[Any]
    summary: Dict[str, Any]

    def __post_init__(self):
        self._samples_df = None
        self._marks_df = None

    @property
    def samples_df(self):
        if self._samples_df is None:
            try:
                import pandas as pd
            except Exception as e:
                raise RuntimeError("pandas is required for plotting/reporting") from e

            rows = []
            for s in self.samples or []:
                rows.append({
                    "t": float(getattr(s, "t", 0.0)),
                    "p_w": getattr(s, "p_w", None),
                    "gpu_util": getattr(s, "gpu_util", None),
                    "mem_util": getattr(s, "mem_util", None),
                })
            df = pd.DataFrame(rows)
            if not df.empty:
                t0 = float(df["t"].min())
                df["ts"] = df["t"] - t0
            self._samples_df = df
        return self._samples_df

    @property
    def marks_df(self):
        if self._marks_df is None:
            try:
                import pandas as pd
            except Exception as e:
                raise RuntimeError("pandas is required for plotting/reporting") from e

            rows = []
            for m in self.marks or []:
                rows.append({
                    "name": str(getattr(m, "name", "")),
                    "t": float(getattr(m, "t", 0.0)),
                    "meta": json.dumps(getattr(m, "meta", {}) or {}),
                })
            self._marks_df = pd.DataFrame(rows)
        return self._marks_df

    def list_figures(self) -> List[str]:
        from .plots import list_figures
        return list_figures()

    def plot_one(
        self,
        fig: str,
        *,
        out_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 300,
        style: str = "science",
        **opts,
    ) -> bool:
        """Plot a single figure by key. Returns True if produced, False if skipped."""
        from .plots import PLOT_REGISTRY

        if fig not in PLOT_REGISTRY:
            raise ValueError(f"Unknown figure '{fig}'. Available: {', '.join(self.list_figures())}")

        spec = PLOT_REGISTRY[fig]
        import matplotlib.pyplot as plt

        _apply_publication_style(style)

        # Paper-friendly defaults without forcing a specific color palette.
        figsize = opts.pop("figsize", (7.2, 4.2))
        fig_obj, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.grid(True)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.8)

        ok = bool(spec.fn(self, ax, **opts))
        if ok:
            title = textwrap.fill(spec.title, width=34)
            ax.set_title(title, pad=12, fontsize=13)
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)

                # PNG ONLY
                out_path_png = os.path.join(out_dir, f"{spec.key}.png")
                fig_obj.savefig(out_path_png, dpi=dpi, format="png")

            # show is kept for API compatibility, but with Agg backend it won't block.
            if show:
                try:
                    plt.show()
                except Exception:
                    pass

        plt.close(fig_obj)
        return ok

    def plot(
        self,
        *,
        figs: Optional[Sequence[str]] = None,
        all: bool = False,
        out_dir: Optional[str] = None,
        show: bool = False,
        **opts,
    ) -> Dict[str, bool]:
        """Plot selected figures (or all=True). Returns per-figure status."""
        keys = list(self.list_figures()) if all else list(figs or [])
        if not keys:
            raise ValueError("No figures requested. Use all=True or provide figs=[...]")
        out: Dict[str, bool] = {}
        for k in keys:
            try:
                out[k] = self.plot_one(k, out_dir=out_dir, show=show, **opts)
            except Exception:
                # Keep going: missing columns etc.
                out[k] = False
        return out

    def export(
        self,
        out_dir: str,
        *,
        include_csv: bool = True,
        include_summary: bool = True,
        include_figures: bool = True,
        **plot_opts,
    ):
        """Optional persistence for reproducibility."""
        os.makedirs(out_dir, exist_ok=True)
        if include_summary:
            with open(os.path.join(out_dir, f"summary_{self.backend}.json"), "w", encoding="utf-8") as f:
                json.dump(self.summary or {}, f, indent=2)

        if include_csv:
            import pandas as pd  # noqa: F401
            self.samples_df.to_csv(os.path.join(out_dir, f"samples_{self.backend}.csv"), index=False)
            self.marks_df.to_csv(os.path.join(out_dir, f"marks_{self.backend}.csv"), index=False)

        if include_figures:
            fig_dir = os.path.join(out_dir, "figures")
            self.plot(all=True, out_dir=fig_dir, show=False, **plot_opts)

        return out_dir

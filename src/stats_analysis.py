# src/stats_analysis.py
from typing import Dict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def describe_per_gauge(gauges: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Basic descriptive stats per gauge (numeric columns).
    """
    descs: Dict[str, pd.DataFrame] = {}
    for gauge_id, df in gauges.items():
        descs[gauge_id] = df.describe(include="number").T
    return descs


def aggregate_describe(gauges: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all gauges and compute global describe() (for rough overview).
    """
    combined = pd.concat(gauges.values(), axis=0, ignore_index=True)
    return combined.describe(include="number").T


def correlation_matrix(gauge_df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """
    Correlation matrix for one gauge.
    """
    return gauge_df.corr(numeric_only=numeric_only)


def plot_histograms(
    gauge_df: pd.DataFrame,
    columns: list[str],
    out_dir: str | Path | None = None,
    gauge_id: str | None = None,
):
    """
    Quick histograms for selected columns of a single gauge.
    """
    for col in columns:
        if col not in gauge_df.columns:
            continue
        plt.figure()
        gauge_df[col].hist(bins=50)
        plt.title(f"Histogram of {col} ({gauge_id})" if gauge_id else f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{gauge_id}_{col}_hist.png" if gauge_id else f"{col}_hist.png"
            plt.savefig(out_dir / fname, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def scatter_plots(
    gauge_df: pd.DataFrame,
    features: list[str],
    target: str,
    out_dir: str | Path | None = None,
    gauge_id: str | None = None,
):
    """
    Quick scatter plots for selected features vs target of a single gauge.
    """

    for feat in features:
        plt.figure()
        plt.scatter(gauge_df[feat], gauge_df[target], alpha=0.3)
        plt.title(f"{feat} vs {target} ({gauge_id})")
        plt.xlabel(feat)
        plt.ylabel(target)
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{gauge_id}_{feat}_hist.png" if gauge_id else f"{feat}_scat.png"
            plt.savefig(out_dir / fname, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
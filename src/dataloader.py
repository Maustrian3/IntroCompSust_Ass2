# src/data_loader.py
from pathlib import Path
import random
from typing import Dict
import pandas as pd


def list_gauge_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("*.csv"))


def load_random_gauges(
    data_dir: str | Path,
    n_samples: int = 100,
    seed: int | None = 42,
    sep: str = ";"
) -> Dict[str, pd.DataFrame]:
    """
    Load n_samples random gauge CSVs from data_dir.
    Returns dict: {gauge_id (filename stem): DataFrame}
    """
    data_dir = Path(data_dir)
    files = list_gauge_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if seed is not None:
        random.seed(seed)

    sampled_files = random.sample(files, min(n_samples, len(files)))
    gauges: Dict[str, pd.DataFrame] = {}

    for f in sampled_files:
        gauge_id = f.stem
        df = pd.read_csv(f, sep=sep)

        if all(col in df.columns for col in ["YYYY", "MM", "DD"]):
            df["date"] = pd.to_datetime(
                df[["YYYY", "MM", "DD"]].rename(
                    columns={"YYYY": "year", "MM": "month", "DD": "day"}
                )
            )

        gauges[gauge_id] = df

    return gauges

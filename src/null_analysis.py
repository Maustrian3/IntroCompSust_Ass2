# src/null_analysis.py
from typing import Dict
import pandas as pd


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a datetime 'date' column.
    Assumes LamaH-style columns YYYY, MM, DD if 'date' is missing.
    """
    if "date" in df.columns:
        return df

    if all(col in df.columns for col in ["YYYY", "MM", "DD"]):
        df = df.copy()
        df["date"] = pd.to_datetime(
            {
                "year": df["YYYY"],
                "month": df["MM"],
                "day": df["DD"],
            }
        )
        return df

    raise ValueError("No 'date' column and no YYYY/MM/DD columns found.")


def gauge_null_and_missing_days(
    gauges: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Return a table with:
      - n_null_values: total NaNs in the gauge DataFrame
      - n_missing_days: number of missing calendar days between first and last date
      - n_days_present: number of unique dates present
      - start_date, end_date: min/max date per gauge
    """
    rows = []

    for gauge_id, df in gauges.items():
        df = _ensure_date_column(df)

        # count all nulls in the whole DataFrame
        n_null = int(df.isna().sum().sum())

        # work with unique sorted dates
        dates = pd.to_datetime(df["date"])
        dates_unique = dates.dropna().sort_values().unique()

        if len(dates_unique) > 0:
            start_date = dates_unique[0]
            end_date = dates_unique[-1]

            full_range = pd.date_range(start_date, end_date, freq="D")
            n_missing_days = len(full_range) - len(dates_unique)
        else:
            start_date = None
            end_date = None
            n_missing_days = None

        rows.append(
            {
                "gauge_id": gauge_id,
                "n_null_values": n_null,
                "n_missing_days": n_missing_days,
                "n_days_present": len(dates_unique),
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    summary = pd.DataFrame(rows).set_index("gauge_id")
    return summary

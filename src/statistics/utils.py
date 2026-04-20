from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_state_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def get_topic_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c.startswith("topic_")]


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    if p_values.empty:
        return p_values.copy()

    ranked = p_values.sort_values()
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]
    adjusted = adjusted.clip(upper=1.0)
    return adjusted.reindex(p_values.index)

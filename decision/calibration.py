"""Calibrate acoustic thresholds T_jk from candidate song pool."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config_columns import ACOUSTIC_METRICS, DEFAULT_ACOUSTIC_POLICY


def calibrate_acoustic_thresholds(
    df: pd.DataFrame,
    policy: Optional[dict] = None,
    quantile: float = 0.20,
) -> dict[str, dict[str, Optional[float]]]:
    """
    對每檔期、每指標：T = min(政策初值, 候選曲 quantile)。
    確保約 quantile 比例的候選曲單獨即可達標，提高 ILP 可行率。
    """
    policy = policy or DEFAULT_ACOUSTIC_POLICY
    out: dict[str, dict[str, Optional[float]]] = {}

    for period, metrics in policy.items():
        out[period] = {}
        for metric in ACOUSTIC_METRICS:
            pol = metrics.get(metric)
            if pol is None:
                out[period][metric] = None
                continue
            if metric not in df.columns:
                out[period][metric] = None
                continue
            vals = pd.to_numeric(df[metric], errors='coerce').dropna()
            if vals.empty:
                out[period][metric] = float(pol)
                continue
            qv = float(np.quantile(vals, quantile))
            out[period][metric] = round(min(float(pol), qv), 4)
    return out

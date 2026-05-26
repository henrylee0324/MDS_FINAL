"""Per-song cost assignment (business layer, independent of y_hat tier)."""
from __future__ import annotations

import pandas as pd

from config_columns import COST_BY_PROFILE


def relative_flop_hop_top(y: pd.Series) -> pd.Series:
    """候選池內相對三分位（簡報用，避免全 TOP）。"""
    q33, q66 = y.quantile(0.33), y.quantile(0.66)
    return y.apply(
        lambda v: 'TOP' if v >= q66 else ('HOP' if v >= q33 else 'FLOP')
    )


def assign_profile_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    costs = [float(COST_BY_PROFILE.get(r['profile_name'], 35.0)) for _, r in out.iterrows()]
    out['cost'] = costs
    if 'y_hat' in out.columns:
        out['tier'] = relative_flop_hop_top(out['y_hat'])
    else:
        out['tier'] = 'HOP'
    return out

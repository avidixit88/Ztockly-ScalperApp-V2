"""
Time-of-day helpers for US equities scalping.

We classify a bar timestamp into:
- OPENING: 09:30–11:00 ET (first 90 minutes)
- MIDDAY:  11:00–15:00 ET (chop zone)
- POWER:   15:00–16:00 ET (power hour)
- OFF:     outside regular session
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


ET = "America/New_York"


def classify_session(ts: pd.Timestamp) -> str:
    if ts is None:
        return "OFF"
    if ts.tzinfo is None:
        # Alpha Vantage timestamps often come as naive; treat as ET for practical trading use
        ts = ts.tz_localize(ET)
    else:
        ts = ts.tz_convert(ET)

    t = ts.time()
    # Regular hours
    if t < pd.Timestamp("09:30", tz=ET).time() or t > pd.Timestamp("16:00", tz=ET).time():
        return "OFF"
    if t < pd.Timestamp("11:00", tz=ET).time():
        return "OPENING"
    if t < pd.Timestamp("15:00", tz=ET).time():
        return "MIDDAY"
    return "POWER"

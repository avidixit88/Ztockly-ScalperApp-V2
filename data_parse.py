from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


def parse_intraday_ohlcv(payload: Dict[str, Any]) -> pd.DataFrame:
    ts_key = None
    for k in payload.keys():
        if k.lower().startswith("time series"):
            ts_key = k
            break
    if not ts_key:
        raise ValueError(f"Could not find time series in payload keys: {list(payload.keys())}")

    ts = payload[ts_key]
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    })
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df


def parse_global_quote(payload: Dict[str, Any]) -> Optional[float]:
    q = payload.get("Global Quote") or payload.get("Global quote") or payload.get("global quote")
    if not isinstance(q, dict):
        return None
    for key in ["05. price", "5. price", "price"]:
        if key in q:
            try:
                return float(q[key])
            except Exception:
                return None
    return None


def parse_indicator(payload: Dict[str, Any], value_key: Optional[str] = None) -> pd.DataFrame:
    ta_key = None
    for k in payload.keys():
        if k.lower().startswith("technical analysis"):
            ta_key = k
            break
    if not ta_key:
        raise ValueError(f"Could not find technical analysis in payload keys: {list(payload.keys())}")

    ta = payload[ta_key]
    df = pd.DataFrame.from_dict(ta, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)

    if value_key is None:
        if len(df.columns) != 1:
            return df.astype(float)
        value_key = df.columns[0]
    return df[[value_key]].astype(float)

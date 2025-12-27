from __future__ import annotations

from typing import List, Tuple, Optional
import pandas as pd

from av_client import AlphaVantageClient
from data_parse import parse_intraday_ohlcv, parse_indicator, parse_global_quote
from signals import compute_scalp_signal, SignalResult


def fetch_bundle(client: AlphaVantageClient, symbol: str, interval: str = "1min") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, Optional[float]]:
    intraday = client.time_series_intraday(symbol, interval=interval, outputsize="compact")
    ohlcv = parse_intraday_ohlcv(intraday)

    rsi5 = client.rsi(symbol, interval=interval, time_period=5)
    rsi14 = client.rsi(symbol, interval=interval, time_period=14)
    macd = client.macd(symbol, interval=interval)

    rsi5_df = parse_indicator(rsi5)
    rsi14_df = parse_indicator(rsi14)
    macd_df = parse_indicator(macd)

    quote_payload = client.quote(symbol)
    last_price = parse_global_quote(quote_payload)

    hist_col = None
    for c in macd_df.columns:
        if "hist" in c.lower():
            hist_col = c
            break
    if hist_col is None:
        hist_col = macd_df.columns[-1]

    return (
        ohlcv,
        rsi5_df.iloc[:, 0],
        rsi14_df.iloc[:, 0],
        macd_df[hist_col],
        last_price,
    )


def scan_watchlist(
    client: AlphaVantageClient,
    symbols: List[str],
    *,
    interval: str = "1min",
    mode: str = "Cleaner signals",
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
) -> List[SignalResult]:
    results: List[SignalResult] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            ohlcv, rsi5, rsi14, macd_hist, quote = fetch_bundle(client, sym, interval=interval)
            res = compute_scalp_signal(
                sym, ohlcv, rsi5, rsi14, macd_hist,
                mode=mode,
                pro_mode=pro_mode,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
            )
            if quote is not None:
                res.last_price = float(quote)
            results.append(res)
        except Exception as e:
            results.append(SignalResult(sym, "NEUTRAL", 0, f"Fetch/error: {e}", None, None, None, None, None, None, "OFF", {}))
    results.sort(key=lambda r: (r.bias in ["LONG", "SHORT"], r.setup_score), reverse=True)
    return results

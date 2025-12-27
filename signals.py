from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from indicators import (
    vwap as calc_vwap,
    atr as calc_atr,
    ema as calc_ema,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    in_zone,
)
from sessions import classify_session


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/OFF
    extras: Dict[str, Any]         # Pro diagnostics


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    lookback_bars: int = 180,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = ohlcv.copy().tail(int(lookback_bars)).copy()
    df["vwap"] = calc_vwap(df)
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap = df["vwap"]
    atr14 = df["atr14"]
    ema20 = df["ema20"]
    ema50 = df["ema50"]

    last_ts = df.index[-1]
    session = classify_session(last_ts)

    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
    )
    last_price = float(close.iloc[-1])

    if not allowed:
        return SignalResult(symbol, "NEUTRAL", 0, f"Filtered by time-of-day ({session})", None, None, None, None, last_price, last_ts, session, {})

    # BASIC events
    was_below_vwap = (close.shift(3) < vwap.shift(3)).iloc[-1] or (close.shift(5) < vwap.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap.shift(3)).iloc[-1] or (close.shift(5) > vwap.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap.shift(1).iloc[-1])

    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings for stops + liquidity reference
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    atr_last = float(atr14.iloc[-1]) if np.isfinite(atr14.iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0

    extras: Dict[str, Any] = {}

    # Pro diagnostics
    trend_long_ok = bool((close.iloc[-1] >= ema20.iloc[-1]) and (ema20.iloc[-1] >= ema50.iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= ema20.iloc[-1]) and (ema20.iloc[-1] <= ema50.iloc[-1]))
    extras["ema20"] = float(ema20.iloc[-1])
    extras["ema50"] = float(ema50.iloc[-1])
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())
    bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low))
    bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high))
    extras["prior_swing_high"] = prior_swing_high
    extras["prior_swing_low"] = prior_swing_low
    extras["bull_liquidity_sweep"] = bull_sweep
    extras["bear_liquidity_sweep"] = bear_sweep

    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, atr14, side="bull", lookback=35)
    ob_bear = find_order_block(df, atr14, side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear

    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    last_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    displacement = bool(atr_last and last_range >= 1.5 * atr_last)
    extras["displacement"] = displacement
    extras["atr14"] = atr_last

    # Scoring
    long_points = 0
    long_reasons = []
    if was_below_vwap and reclaim_vwap:
        long_points += 35; long_reasons.append("VWAP reclaim")
    if rsi_snap and rsi14 < 60:
        long_points += 20; long_reasons.append("RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        long_points += 20; long_reasons.append("MACD hist turning up")
    if vol_ok:
        long_points += 15; long_reasons.append("Volume confirmation")
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        long_points += 10; long_reasons.append("Higher-low micro structure")

    short_points = 0
    short_reasons = []
    if was_above_vwap and reject_vwap:
        short_points += 35; short_reasons.append("VWAP rejection")
    if rsi_downshift and rsi14 > 40:
        short_points += 20; short_reasons.append("RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        short_points += 20; short_reasons.append("MACD hist turning down")
    if vol_ok:
        short_points += 15; short_reasons.append("Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        short_points += 10; short_reasons.append("Lower-high micro structure")

    if pro_mode:
        if bull_sweep:
            long_points += 20; long_reasons.append("Liquidity sweep (low)")
        if bear_sweep:
            short_points += 20; short_reasons.append("Liquidity sweep (high)")
        if bull_ob_retest:
            long_points += 15; long_reasons.append("Bullish order block retest")
        if bear_ob_retest:
            short_points += 15; short_reasons.append("Bearish order block retest")
        if bull_fvg is not None:
            long_points += 10; long_reasons.append("Bullish FVG present")
        if bear_fvg is not None:
            short_points += 10; short_reasons.append("Bearish FVG present")
        if displacement:
            long_points += 5; short_points += 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # Requirements
    if int(cfg["require_vwap_event"]) == 1:
        if not ((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap)):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_rsi_event"]) == 1:
        if not (rsi_snap or rsi_downshift):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_macd_turn"]) == 1:
        if not (macd_turn_up or macd_turn_down):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)
    if int(cfg["require_volume"]) == 1 and not vol_ok:
        return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "No volume confirmation", None, None, None, None, last_price, last_ts, session, extras)

    if pro_mode:
        if not (bull_sweep or bear_sweep or bull_ob_retest or bear_ob_retest):
            return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), "Pro mode: no liquidity sweep / OB retest trigger", None, None, None, None, last_price, last_ts, session, extras)

    min_score = int(cfg["min_actionable_score"])

    if long_points >= min_score and long_points > short_points:
        entry = last_price
        stop = float(min(recent_swing_low, last_price - max(atr_last, 0.0) * 0.8))
        risk = max(entry - stop, 0.01)
        return SignalResult(symbol, "LONG", min(100, int(long_points)), ", ".join(long_reasons[:6]), entry, stop, entry + risk, entry + 2 * risk, last_price, last_ts, session, extras)

    if short_points >= min_score and short_points > long_points:
        entry = last_price
        stop = float(max(recent_swing_high, last_price + max(atr_last, 0.0) * 0.8))
        risk = max(stop - entry, 0.01)
        return SignalResult(symbol, "SHORT", min(100, int(short_points)), ", ".join(short_reasons[:6]), entry, stop, entry - risk, entry - 2 * risk, last_price, last_ts, session, extras)

    reason = f"LongScore={long_points} ({', '.join(long_reasons)}); ShortScore={short_points} ({', '.join(short_reasons)})"
    return SignalResult(symbol, "NEUTRAL", int(max(long_points, short_points)), reason, None, None, None, None, last_price, last_ts, session, extras)

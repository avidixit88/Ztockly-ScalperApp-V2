"""
Alpha Vantage client helpers for intraday scalping dashboards.

Set env var: ALPHAVANTAGE_API_KEY

Optional (premium plans): entitlement=realtime
"""
from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional

BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class AVConfig:
    api_key: str
    entitlement: Optional[str] = "realtime"
    min_seconds_between_calls: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 20


class AlphaVantageClient:
    def __init__(self, config: Optional[AVConfig] = None):
        if config is None:
            api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "Missing ALPHAVANTAGE_API_KEY env var. "
                    "Set it before running."
                )
            config = AVConfig(api_key=api_key)

        self.cfg = config
        self._last_call_ts = 0.0

    def _pace(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.cfg.min_seconds_between_calls:
            time.sleep(self.cfg.min_seconds_between_calls - elapsed)

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params["apikey"] = self.cfg.api_key
        if self.cfg.entitlement:
            params["entitlement"] = self.cfg.entitlement

        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._pace()
                r = requests.get(BASE_URL, params=params, timeout=self.cfg.timeout_seconds)
                self._last_call_ts = time.time()
                r.raise_for_status()
                data = r.json()

                if isinstance(data, dict) and any(k in data for k in ["Error Message", "Information", "Note"]):
                    msg = data.get("Error Message") or data.get("Information") or data.get("Note")
                    raise RuntimeError(f"Alpha Vantage response warning/error: {msg}")

                return data
            except Exception as e:
                last_err = e
                time.sleep(min(2.5 * attempt, 8.0))
        raise RuntimeError(f"Alpha Vantage request failed after retries: {last_err}")

    def time_series_intraday(self, symbol: str, interval: str = "1min", outputsize: str = "compact") -> Dict[str, Any]:
        return self._request({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
        })

    def quote(self, symbol: str) -> Dict[str, Any]:
        return self._request({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        })

    def rsi(self, symbol: str, interval: str = "1min", time_period: int = 14, series_type: str = "close") -> Dict[str, Any]:
        return self._request({
            "function": "RSI",
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
        })

    def macd(
        self,
        symbol: str,
        interval: str = "1min",
        series_type: str = "close",
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> Dict[str, Any]:
        return self._request({
            "function": "MACD",
            "symbol": symbol,
            "interval": interval,
            "series_type": series_type,
            "fastperiod": fastperiod,
            "slowperiod": slowperiod,
            "signalperiod": signalperiod,
        })

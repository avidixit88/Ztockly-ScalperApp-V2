# Ztockly Scalping Scanner v4 (Basic + Pro Mode)

- Ranked watchlist scanner
- Modes: **Fast scalp** / **Cleaner signals**
- Time-of-day filters (ET) + cooldown + in-app alerts
- **Pro mode toggle** adds:
  - Liquidity sweep detection
  - Order block detection + retest proximity
  - Fair Value Gap (FVG) detection
  - EMA20/EMA50 trend context

## Run
```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
streamlit run app.py
```

## Alerts
Alerts are stored inside Streamlit session state (no email/webhook).

## Pro mode
Pro mode is more selective and requires at least one structure trigger:
- liquidity sweep OR order-block retest

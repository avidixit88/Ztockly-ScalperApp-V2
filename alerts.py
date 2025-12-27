"""
Alerting helpers.

We support:
- Generic webhook POST (Slack/Discord/your server/Twilio proxy)
- Optional SendGrid email (if you provide SENDGRID_API_KEY)

We intentionally DO NOT bake in your personal email credentials.

Env vars (optional):
- ALERT_WEBHOOK_URL: a URL to POST JSON to
- SENDGRID_API_KEY: if using SendGrid
- ALERT_EMAIL_TO: recipient email (e.g., avidixit@gmail.com)
- ALERT_EMAIL_FROM: verified sender email for SendGrid
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional
import requests


def post_webhook(payload: Dict[str, Any], url: Optional[str] = None, timeout: int = 10) -> bool:
    url = (url or os.getenv("ALERT_WEBHOOK_URL", "")).strip()
    if not url:
        return False
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code >= 200 and r.status_code < 300
    except Exception:
        return False


def send_sendgrid_email(subject: str, content: str) -> bool:
    api_key = os.getenv("SENDGRID_API_KEY", "").strip()
    to_email = os.getenv("ALERT_EMAIL_TO", "").strip()
    from_email = os.getenv("ALERT_EMAIL_FROM", "").strip()
    if not (api_key and to_email and from_email):
        return False

    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": content}],
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
        return r.status_code in (200, 202)
    except Exception:
        return False

"""
Minimal helper for sending alerts to Microsoft Teams via an Incoming
Webhook connector, used by both the ingestion and training DAGs.

Configure the webhook URLs as Airflow Variables:
  - TEAMS_WEBHOOK_DATA_QUALITY  -> "Data Quality Alerts" channel
  - TEAMS_WEBHOOK_ML_ALERTS     -> "ML Alerts" channel

If a webhook Variable isn't set, the alert is logged and skipped instead
of raising, so a missing webhook never breaks the DAG run.
"""
import logging

import requests

logger = logging.getLogger(__name__)


def send_teams_alert(webhook_url: str, title: str, text: str, color: str = "FF0000"):
    if not webhook_url:
        logger.warning(
            "Teams webhook URL not configured - skipping alert: %s", title
        )
        return

    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": color,
        "title": title,
        "text": text,
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Teams alert sent: %s", title)
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to send Teams alert '%s': %s", title, exc)

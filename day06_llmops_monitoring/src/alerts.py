# alerts.py — simple console/email placeholder for alerting
from __future__ import annotations
def alert_if_drift(drift:bool, threshold:float=0.3):
    if drift:
        print("🚨 ALERT: Data drift detected! Notify team / trigger retraining.")
    else:
        print("✅ No significant drift detected.")
    # Placeholder: integrate with email/SMS/Slack APIs as needed
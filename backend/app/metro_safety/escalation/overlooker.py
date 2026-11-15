"""Overlooker Alert & Notification Module.
Dispatches critical safety notifications to station oversight personnel via:
- Dashboard Alert Banner & Snapshot
- Windows Desktop Toast Notifications
- Optional Outbound Webhook (ntfy.sh and/or Telegram bot)
- Enforces 10-second per-event-type cooldown and persistent audit logging.
"""
import os
import sys
import time
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("metro_safety.escalation.overlooker")

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


class OverlookerNotifier:
    """Manages notifications for high-severity platform events (L2, L3, DISTRESS)."""

    def __init__(
        self,
        cooldown_sec: float = 10.0,
        desktop_notification: bool = True,
        audit_log_file: str = "logs/overlooker_audit.log",
        ntfy_config: Optional[Dict[str, Any]] = None,
        telegram_config: Optional[Dict[str, Any]] = None,
        dashboard_hub=None
    ):
        self.cooldown_sec = float(cooldown_sec)
        self.desktop_notification = desktop_notification
        self.audit_log_path = Path(audit_log_file)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.ntfy_config = ntfy_config or {"enabled": False, "topic": "", "server": "https://ntfy.sh"}
        self.telegram_config = telegram_config or {"enabled": False, "bot_token": "", "chat_id": ""}
        self.dashboard_hub = dashboard_hub

        self._last_sent_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def should_notify(self, severity: str, event_type: str) -> Tuple[bool, str]:
        """Check if event meets severity threshold and passes cooldown."""
        sev = str(severity).upper().strip()
        # Only notify overlooker for L2, L3, and DISTRESS
        if sev not in ["L2", "L3", "DISTRESS"]:
            return False, f"Severity '{sev}' does not meet overlooker threshold (L2/L3/DISTRESS required)"

        with self._lock:
            now = time.time()
            last_time = self._last_sent_times.get(event_type, 0.0)
            elapsed = now - last_time
            if elapsed < self.cooldown_sec:
                remaining = self.cooldown_sec - elapsed
                return False, f"Cooldown active for '{event_type}' ({remaining:.1f}s remaining)"

            # Update cooldown timestamp
            self._last_sent_times[event_type] = now
            return True, "OK"

    def notify(
        self,
        event_type: str,
        severity: str,
        track_id: int,
        zone: str,
        confidence: float,
        timestamp: float,
        snapshot_base64: Optional[str] = None,
        train_state: str = "ABSENT"
    ) -> Dict[str, Any]:
        """Attempt to dispatch an overlooker alert across all configured channels."""
        can_notify, reason = self.should_notify(severity, event_type)
        if not can_notify:
            logger.debug(f"[Overlooker] Suppressed: {reason}")
            return {"status": "suppressed", "reason": reason}

        title = f"DMRC SAFETY ALERT: {severity} - {event_type}"
        is_distress = (severity.upper() == "DISTRESS")

        if is_distress:
            body = (
                f"[SILENT ALERT] Distress gesture verified on platform.\n"
                f"Track ID: {track_id} | Zone: {zone}\n"
                f"ACTION: Dispatch security to platform silently. Audible buzzer suppressed."
            )
        else:
            body = (
                f"Incident: {event_type} | Severity: {severity}\n"
                f"Track ID: {track_id} | Zone: {zone} | Train: {train_state}\n"
                f"Confidence: {confidence:.2f}"
            )

        channels_dispatched = []

        # 1. Dashboard Alert Banner
        if self.dashboard_hub is not None:
            try:
                self.dashboard_hub.set_overlooker_alert({
                    "title": title,
                    "body": body,
                    "severity": severity,
                    "event_type": event_type,
                    "track_id": track_id,
                    "zone": zone,
                    "timestamp": timestamp,
                    "snapshot": snapshot_base64
                })
                channels_dispatched.append("dashboard_banner")
            except Exception as e:
                logger.error(f"Dashboard alert dispatch failed: {e}")

        # 2. Windows Desktop Notification (Toast)
        if self.desktop_notification and sys.platform == "win32":
            threading.Thread(
                target=self._send_windows_toast,
                args=(title, body),
                daemon=True
            ).start()
            channels_dispatched.append("windows_toast")

        # 3. Optional Outbound Push (ntfy.sh)
        if self.ntfy_config.get("enabled") and self.ntfy_config.get("topic"):
            threading.Thread(
                target=self._send_ntfy_push,
                args=(title, body, severity),
                daemon=True
            ).start()
            channels_dispatched.append("ntfy_push")

        # 4. Optional Outbound Push (Telegram)
        if self.telegram_config.get("enabled") and self.telegram_config.get("bot_token") and self.telegram_config.get("chat_id"):
            threading.Thread(
                target=self._send_telegram_push,
                args=(f"🚨 *{title}*\n\n{body}",),
                daemon=True
            ).start()
            channels_dispatched.append("telegram_push")

        # 5. Persistent Audit Logging
        audit_record = {
            "timestamp": timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "event_type": event_type,
            "severity": severity,
            "track_id": track_id,
            "zone": zone,
            "confidence": round(confidence, 4),
            "train_state": train_state,
            "channels_dispatched": channels_dispatched
        }
        self._write_audit_log(audit_record)

        logger.info(f"[Overlooker Alert Dispatched] {title} via {channels_dispatched}")
        return {
            "status": "dispatched",
            "channels": channels_dispatched,
            "record": audit_record
        }

    def _write_audit_log(self, record: Dict[str, Any]):
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write overlooker audit log: {e}")

    def _send_windows_toast(self, title: str, body: str):
        """Dispatch a native Windows toast notification via PowerShell."""
        clean_title = title.replace("'", "").replace('"', "")
        clean_body = body.replace("'", "").replace('"', "").replace("\n", " ")

        ps_cmd = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textNodes = $template.GetElementsByTagName('text')
$textNodes.Item(0).AppendChild($template.CreateTextNode('{clean_title}')) | Out-Null
$textNodes.Item(1).AppendChild($template.CreateTextNode('{clean_body}')) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('DMRC Platform Safety').Show($toast)
"""
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                capture_output=True,
                timeout=3.0,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
        except Exception as e:
            logger.debug(f"Windows toast dispatch failed (non-critical): {e}")

    def _send_ntfy_push(self, title: str, body: str, severity: str):
        """Send optional push notification to ntfy.sh topic."""
        try:
            server = self.ntfy_config.get("server", "https://ntfy.sh").rstrip("/")
            topic = self.ntfy_config.get("topic")
            url = f"{server}/{topic}"

            priority = "5" if severity == "L3" else ("4" if severity == "L2" else "3")
            tags = "warning,rotating_light" if severity == "L3" else "warning,metro"

            req = urllib.request.Request(
                url,
                data=body.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": priority,
                    "Tags": tags
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                logger.info(f"ntfy.sh notification response status: {resp.status}")
        except Exception as e:
            logger.debug(f"Optional ntfy.sh dispatch failed (non-critical): {e}")

    def _send_telegram_push(self, text: str):
        """Send optional push notification to Telegram bot."""
        try:
            token = self.telegram_config.get("bot_token")
            chat_id = self.telegram_config.get("chat_id")
            url = f"https://api.telegram.org/bot{token}/sendMessage"

            data = json.dumps({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                logger.info(f"Telegram notification response status: {resp.status}")
        except Exception as e:
            logger.debug(f"Optional Telegram dispatch failed (non-critical): {e}")


"""Hardware Buzzer Module for Platform Safety System.
Generates audible alarms on the host machine using winsound (Windows) with cross-platform fallbacks.
Runs in a background worker thread so the computer vision pipeline is never blocked.
"""
import sys
import time
import threading
import logging
import io
import math
import struct
import wave
from typing import Optional, Dict, Any

from metro_safety.escalation.adapters import BasePlatformBuzzer

logger = logging.getLogger("metro_safety.escalation.buzzer")

# Native Windows sound support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class HardwareBuzzer(BasePlatformBuzzer):
    """Manages audible buzzer alerts with severity-specific sound patterns,
    non-blocking daemon audio playback, automatic clear-timeout, and operator acknowledge.
    Implements BasePlatformBuzzer for seamless integration with EscalationEngine.
    """

    PATTERNS = {
        "OFF": 0,
        "L1": 1,
        "L2": 2,
        "L3": 3,
    }

    @staticmethod
    def _generate_wav(frequency: int, duration_ms: int) -> bytes:
        sample_rate = 22050
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            fade_len = min(int(sample_rate * 0.005), num_samples // 4) if num_samples > 10 else 1
            data = bytearray()
            for i in range(num_samples):
                gain = 1.0
                if i < fade_len:
                    gain = i / float(fade_len)
                elif i > num_samples - fade_len:
                    gain = (num_samples - i) / float(fade_len)
                sample = int(32767 * 0.55 * gain * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                data.extend(struct.pack('<h', sample))
            wav.writeframes(data)
        return buf.getvalue()

    def __init__(
        self,
        audio_enabled: bool = True,
        clear_timeout_sec: float = 4.0,
        l1_freq: int = 800,
        l1_dur_ms: int = 150,
        l2_freq: int = 1200,
        l2_beep_ms: int = 200,
        l2_pause_ms: int = 100,
        l3_low_freq: int = 1800,
        l3_high_freq: int = 2200,
        beep_fn=None
    ):
        self.audio_enabled = audio_enabled
        self.clear_timeout_sec = clear_timeout_sec
        self.l1_freq = l1_freq
        self.l1_dur_ms = l1_dur_ms
        self.l2_freq = l2_freq
        self.l2_beep_ms = l2_beep_ms
        self.l2_pause_ms = l2_pause_ms
        self.l3_low_freq = l3_low_freq
        self.l3_high_freq = l3_high_freq

        # Allow dependency injection of beep function for unit testing
        self._beep_fn = beep_fn if beep_fn is not None else self._default_beep

        self._muted = False
        self._current_pattern = "OFF"
        self._last_trigger_time = 0.0
        self._last_test_pattern: Optional[str] = None
        self._last_test_time: float = 0.0
        self._acknowledged = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        # Start background playback daemon thread
        self._thread = threading.Thread(target=self._audio_loop, daemon=True, name="HardwareBuzzerWorker")
        self._thread.start()

    def _default_beep(self, frequency: int, duration_ms: int):
        """Emit beep using Windows audio, winsound.Beep, or terminal bell fallback."""
        if not self.audio_enabled or self._muted:
            return
        if HAS_WINSOUND:
            # 1. Synthesized audio tone played via standard audio output (works on all modern PCs)
            try:
                wav_bytes = self._generate_wav(frequency, duration_ms)
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
                return
            except Exception as e:
                logger.debug(f"winsound.PlaySound error: {e}")

            # 2. Native winsound.Beep fallback (legacy motherboard speaker)
            try:
                freq = max(37, min(32767, int(frequency)))
                dur = max(20, int(duration_ms))
                winsound.Beep(freq, dur)
                return
            except Exception as e:
                logger.debug(f"winsound.Beep error: {e}")

            # 3. System audio exclamation fallback
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return
            except Exception:
                pass
        else:
            try:
                sys.stdout.write('\a')
                sys.stdout.flush()
                time.sleep(duration_ms / 1000.0)
            except Exception:
                pass

    def activate(self, platform_id: str = "", pattern: str = "PULSED"):
        """Implement BasePlatformBuzzer.activate() for EscalationEngine integration."""
        pat = str(pattern).upper().strip()
        if pat in ["CONTINUOUS_HIGH", "L3"]:
            self.trigger("L3")
        elif pat in ["PULSED_FAST", "L2"]:
            self.trigger("L2")
        elif pat in ["PULSED", "L1"]:
            self.trigger("L1")
        else:
            self.trigger("L1")

    def deactivate(self, platform_id: str = ""):
        """Implement BasePlatformBuzzer.deactivate()."""
        with self._lock:
            self._current_pattern = "OFF"

    def is_active(self, platform_id: str = "") -> bool:
        """Implement BasePlatformBuzzer.is_active()."""
        with self._lock:
            return self._current_pattern != "OFF"

    def trigger(self, severity: str):
        """Trigger or escalate the buzzer.
        - L1: Single pulse (800Hz)
        - L2: Repeating pulse (1200Hz)
        - L3: Continuous urgent siren (1800Hz / 2200Hz)
        - DISTRESS: Silent alert (NO BUZZER)
        """
        sev = str(severity).upper().strip()
        if sev == "DISTRESS" or sev == "NORMAL":
            return

        if sev not in self.PATTERNS:
            return

        with self._lock:
            curr_priority = self.PATTERNS.get(self._current_pattern, 0)
            new_priority = self.PATTERNS.get(sev, 0)

            # Update if equal or higher severity, or if previously acknowledged and new event occurs
            if new_priority >= curr_priority or self._acknowledged:
                self._current_pattern = sev
                self._acknowledged = False

            self._last_trigger_time = time.time()
            self._wake_event.set()

    def update_clear_status(self, is_intruding: bool):
        """Periodic tick called by the video pipeline:
        If intrusion has cleared for >= clear_timeout_sec, automatically turn off the buzzer.
        """
        with self._lock:
            if is_intruding:
                self._last_trigger_time = time.time()
            elif self._current_pattern != "OFF":
                elapsed = time.time() - self._last_trigger_time
                if elapsed >= self.clear_timeout_sec:
                    self._current_pattern = "OFF"
                    logger.info(f"Buzzer cleared: zone clear for {elapsed:.1f}s >= {self.clear_timeout_sec}s")

    def acknowledge(self):
        """Operator pressed 'Acknowledge' on the dashboard. Silences active buzzer immediately."""
        with self._lock:
            self._acknowledged = True
            self._current_pattern = "OFF"
            logger.info("Buzzer acknowledged by operator - silenced.")

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns new muted state."""
        with self._lock:
            self._muted = not self._muted
            logger.info(f"Buzzer mute toggled: {self._muted}")
            return self._muted

    def set_mute(self, muted: bool):
        with self._lock:
            self._muted = bool(muted)

    def test_beep(self, pattern: str = "L1"):
        """Run a short audible test pattern."""
        pat = str(pattern).upper().strip()
        with self._lock:
            self._last_test_pattern = pat
            self._last_test_time = time.time()

        def _test_worker():
            if pat == "L1":
                self._beep_fn(self.l1_freq, self.l1_dur_ms)
            elif pat == "L2":
                self._beep_fn(self.l2_freq, self.l2_beep_ms)
                time.sleep(0.08)
                self._beep_fn(self.l2_freq, self.l2_beep_ms)
            elif pat == "L3":
                self._beep_fn(self.l3_low_freq, 120)
                self._beep_fn(self.l3_high_freq, 120)
                self._beep_fn(self.l3_low_freq, 120)
        t = threading.Thread(target=_test_worker, daemon=True)
        t.start()

    def stop(self):
        """Permanently stop worker thread on shutdown."""
        self._stop_event.set()
        self._wake_event.set()
        with self._lock:
            self._current_pattern = "OFF"

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            elapsed_clear = time.time() - self._last_trigger_time if self._last_trigger_time > 0 else 999.0
            return {
                "active": self._current_pattern != "OFF",
                "pattern": self._current_pattern,
                "muted": self._muted,
                "audio_enabled": self.audio_enabled,
                "acknowledged": self._acknowledged,
                "last_test_pattern": self._last_test_pattern,
                "last_test_time": self._last_test_time,
                "clear_timer_remaining": max(0.0, self.clear_timeout_sec - elapsed_clear) if self._current_pattern != "OFF" else 0.0
            }

    def _audio_loop(self):
        """Worker loop playing pattern audio while pattern != OFF and not muted."""
        while not self._stop_event.is_set():
            with self._lock:
                pattern = self._current_pattern
                muted = self._muted
                enabled = self.audio_enabled

            if pattern == "OFF" or muted or not enabled:
                self._wake_event.wait(timeout=0.1)
                self._wake_event.clear()
                continue

            if pattern == "L1":
                # L1: Short pulse, then wait briefly
                self._beep_fn(self.l1_freq, self.l1_dur_ms)
                time.sleep(0.5)
            elif pattern == "L2":
                # L2: Repeating double beep
                self._beep_fn(self.l2_freq, self.l2_beep_ms)
                time.sleep(self.l2_pause_ms / 1000.0)
                self._beep_fn(self.l2_freq, self.l2_beep_ms)
                time.sleep(0.3)
            elif pattern == "L3":
                # L3: Continuous urgent wailing siren with interruptible minimum interval
                self._beep_fn(self.l3_low_freq, 150)
                self._beep_fn(self.l3_high_freq, 150)
                self._stop_event.wait(timeout=0.05)


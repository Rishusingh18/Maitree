"""Control-Room Station Dashboard (Flask Web UI).
Connects directly to the LiveHub for:
- Live MJPEG video stream (/video_feed)
- Real-time telemetry (Live FPS, End-to-End Latency, Synthetic Benchmark Accuracy)
- Live Incident Event Log with real-time SSE updates and dynamic table filling
- Operator controls (Buzzer acknowledge, Mute toggle, Test Beep, ETS interlock, Train State)
"""
import time
import json
from pathlib import Path
from flask import Flask, render_template_string, Response, jsonify, request
from metro_safety.demo.hub import LiveHub, get_global_hub

def create_dashboard_app(hub: LiveHub = None) -> Flask:
    app = Flask(__name__)
    active_hub = hub if hub is not None else get_global_hub()

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DMRC Metro Platform Safety System - Central Control Room</title>
    <style>
        :root {
            --bg-main: #0E1117;
            --bg-card: #1A1F2C;
            --bg-subtle: #242B3D;
            --border: #2D3748;
            --text-main: #E2E8F0;
            --text-muted: #94A3B8;
            --accent: #38BDF8;
            --danger: #EF4444;
            --warning: #F59E0B;
            --success: #10B981;
            --purple: #A855F7;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            margin: 0;
            padding: 16px 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }
        .logo-title {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 20px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.5px;
        }
        .header-badges {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-live { background-color: #065F46; color: #6EE7B7; border: 1px solid #059669; }
        .badge-l3 { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid var(--danger); }
        .badge-l2 { background-color: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid var(--warning); }
        .badge-l1 { background-color: rgba(234, 179, 8, 0.2); color: #FEF08A; border: 1px solid #EAB308; }
        .badge-distress { background-color: rgba(168, 85, 247, 0.2); color: #E9D5FF; border: 1px solid var(--purple); }
        .badge-anomaly { background-color: rgba(168, 85, 247, 0.25); color: #D8B4FE; border: 1px solid #C084FC; }
        .badge-secure { background-color: rgba(16, 185, 129, 0.2); color: #A7F3D0; border: 1px solid var(--success); }
        .filter-btn { padding: 3px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; background: var(--bg-subtle); color: var(--text-main); border: 1px solid var(--border); }
        .filter-btn.active { background: var(--accent); color: #000; font-weight: 600; border-color: var(--accent); }

        /* Overlooker Emergency Banner */
        #alert-banner {
            display: none;
            background: linear-gradient(90deg, #991B1B, #7F1D1D);
            border: 2px solid #EF4444;
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 18px;
            animation: pulse-border 1.5s infinite;
        }
        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .alert-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .alert-text h4 { margin: 0 0 4px 0; font-size: 17px; color: #FEE2E2; }
        .alert-text p { margin: 0; font-size: 13px; color: #FCA5A5; }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 18px;
        }
        .card-title {
            font-size: 15px;
            font-weight: 600;
            color: #F1F5F9;
            margin-top: 0;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Video Feed */
        .video-container {
            position: relative;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border);
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .video-overlay-badge {
            position: absolute;
            top: 12px;
            left: 12px;
            z-index: 10;
        }

        /* Telemetry Cards */
        .telemetry-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 13px;
        }
        .telemetry-label { color: var(--text-muted); }
        .telemetry-value { font-weight: 600; font-family: monospace; font-size: 14px; }

        /* Buttons */
        .btn {
            padding: 8px 14px;
            border-radius: 5px;
            border: none;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-danger { background-color: var(--danger); color: white; }
        .btn-danger:hover { background-color: #DC2626; }
        .btn-warning { background-color: var(--warning); color: black; }
        .btn-secondary { background-color: var(--bg-subtle); color: var(--text-main); border: 1px solid var(--border); }
        .btn-secondary:hover { background-color: #333C52; }
        .btn-ack { background-color: #F59E0B; color: #000; font-weight: 700; padding: 10px 18px; }

        /* Table */
        .table-wrap {
            max-height: 290px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: var(--bg-subtle);
            color: var(--accent);
            position: sticky;
            top: 0;
            font-weight: 600;
        }
        tr:hover { background-color: rgba(255,255,255,0.02); }
        .snapshot-thumb {
            width: 36px;
            height: 22px;
            border-radius: 2px;
            cursor: pointer;
            object-fit: cover;
        }

        /* Active train button state (strictly driven by backend train_state) */
        .btn-active-train {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border-color: #60A5FA !important;
            box-shadow: 0 0 10px rgba(37, 99, 235, 0.5) !important;
            font-weight: 700 !important;
        }
        .btn:disabled {
            opacity: 0.45 !important;
            cursor: not-allowed !important;
            pointer-events: none !important;
        }

        /* Error Toast Notifications */
        .toast {
            background: #1E293B;
            color: #FFF;
            border: 1px solid #DC2626;
            border-left: 4px solid #DC2626;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 13px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.5);
            animation: toast-in 0.25s ease-out;
            max-width: 360px;
            pointer-events: auto;
        }
        @keyframes toast-in {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div class="logo-title">
            <span>🚇 DMRC CENTRAL CONTROL ROOM</span>
            <span style="font-size: 13px; color: var(--text-muted); font-weight: normal;">| PLATFORM SAFETY & EMERGENCY INTERLOCK</span>
        </div>
        <div class="header-badges">
            <span class="badge badge-live" id="stream-status">LIVE STREAM</span>
            <span class="badge" style="background: var(--bg-subtle);" id="station-badge">DMRC_STN_042</span>
            <span class="badge" style="background: var(--bg-subtle);" id="cctv-badge">CCTV_CAM_03</span>
            <span class="badge badge-secure" id="train-state-badge">TRAIN: ABSENT</span>
        </div>
    </div>

    <!-- Overlooker Alert Banner -->
    <div id="alert-banner">
        <div class="alert-content">
            <div class="alert-text">
                <h4 id="alert-title">🚨 HIGH SEVERITY INCIDENT DETECTED</h4>
                <p id="alert-body">Track intrusion on platform edge. Traction power cut recommended.</p>
            </div>
            <div>
                <button class="btn btn-ack" onclick="acknowledgeAlarm()">ACKNOWLEDGE & SILENCE BUZZER</button>
            </div>
        </div>
    </div>

    <!-- Adaptive Zones Unverified / Confirmation Banner -->
    <div id="zone-unverified-banner" style="display: none; background: linear-gradient(90deg, #92400E, #78350F); border: 2px solid #F59E0B; border-radius: 8px; padding: 14px 20px; margin-bottom: 18px;">
        <div class="alert-content">
            <div class="alert-text">
                <h4 id="zone-alert-title" style="color: #FEF3C7; margin: 0 0 4px 0; font-size: 16px;">⚠️ ZONES UNVERIFIED — CAMERA DRIFT DETECTED</h4>
                <p id="zone-alert-body" style="color: #FDE68A; margin: 0; font-size: 13px;">Camera misalignment detected. Operator confirmation required to verify zones.</p>
            </div>
            <div>
                <button id="btn-confirm-zones" class="btn btn-ack" style="background: #F59E0B; color: #18181B;" onclick="confirmZoneAlignment()">CONFIRM CURRENT ZONES</button>
            </div>
        </div>
    </div>

    <!-- Main Grid -->
    <div class="grid">
        <!-- Left: Live Video & Event Log -->
        <div>
            <!-- Dead Backend Alert Banner (Shown after 3 failed polls) -->
            <div id="backend-dead-banner" style="display: none; background: #DC2626; color: #FFF; padding: 12px 18px; border-radius: 6px; margin-bottom: 12px; font-weight: 700; font-size: 13px; border: 1px solid #B91C1C; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.4); animation: pulse-border 1.2s infinite;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>⚠️ <strong>BACKEND DISCONNECTED - data is stale</strong></span>
                    <span style="font-size: 11px; background: rgba(0,0,0,0.3); padding: 3px 8px; border-radius: 4px;" id="backend-dead-time">Last update: --</span>
                </div>
            </div>

            <!-- Camera Moving Warning Banner -->
            <div id="camera-motion-banner" style="display: none; background: linear-gradient(90deg, #7C2D12, #C2410C); color: #FFF; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-weight: 700; font-size: 13px; border: 1px solid #EA580C; box-shadow: 0 2px 4px rgba(0,0,0,0.3); animation: pulse-border 1.5s infinite;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>📷 <strong>CAMERA MOVING: needs operator review</strong> (alerts capped at L2)</span>
                    <span style="font-size: 11px; background: rgba(0,0,0,0.3); padding: 3px 8px; border-radius: 4px;" id="camera-motion-reason">Optical flow shake</span>
                </div>
            </div>

            <!-- Prominent Fallback Warning Banner -->
            <div id="fallback-banner" style="display: none; background: linear-gradient(90deg, #92400E, #B45309); color: #FEF3C7; padding: 12px 16px; border-radius: 6px; margin-bottom: 14px; font-weight: 600; font-size: 13px; border: 1px solid #F59E0B; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>⚠️ <strong>SOURCE: DEMO CLIP</strong> (phone stream unreachable)</span>
                    <span style="font-size: 11px; background: rgba(0,0,0,0.3); padding: 3px 8px; border-radius: 4px;" id="fallback-retry">Retrying connection...</span>
                </div>
                <div style="font-size: 11px; color: #FDE68A; margin-top: 4px;" id="fallback-target">Attempted: http://192.168.1.32:8080/video</div>
            </div>

            <div class="card">
                <div class="card-title">
                    <span>Live CCTV Video Feed & Real-time AI Overlays</span>
                    <span style="font-size: 12px; color: var(--text-muted);">Drop Stale: ON | Low-Latency MJPEG</span>
                </div>
                <div class="video-container" style="position: relative;">
                    <img id="live-video" src="/video_feed" alt="CCTV Stream" onload="handleVideoLoad();" onerror="handleVideoError(this);">
                    <!-- Visible Stream Error Overlay -->
                    <div id="video-error-overlay" style="display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(15, 23, 42, 0.92); color: #EF4444; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px; z-index: 25;">
                        <div style="font-size: 32px; margin-bottom: 8px;">⚠️</div>
                        <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">STREAM ERROR / DISCONNECTED</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px;" id="video-error-msg">Waiting for camera pipeline reconnect. Retrying stream...</div>
                    </div>
                    <!-- No Signal Placeholder when Backend is Dead -->
                    <div id="no-signal-placeholder" style="display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #0B0F19; color: #EF4444; flex-direction: column; align-items: center; justify-content: center; text-align: center; z-index: 30; border: 2px dashed #EF4444;">
                        <div style="font-size: 40px; margin-bottom: 8px;">📡</div>
                        <div style="font-size: 18px; font-weight: 800; letter-spacing: 1.5px;">NO SIGNAL</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 6px;" id="no-signal-subtext">Backend server disconnected. Reconnecting automatically...</div>
                    </div>
                    <div class="video-overlay-badge">
                        <span class="badge badge-l3" id="ets-badge">ETS READY</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span>Recent Incident Event Log</span>
                        <div style="display: flex; gap: 4px;" id="event-filter-group">
                            <button class="filter-btn active" id="filter-all" onclick="setEventFilter('all')">All</button>
                            <button class="filter-btn" id="filter-safety" onclick="setEventFilter('safety')">Safety</button>
                            <button class="filter-btn" id="filter-anomaly" onclick="setEventFilter('anomaly')">Anomaly</button>
                        </div>
                    </div>
                    <span style="font-size: 12px; color: var(--text-muted);" id="event-count">0 events</span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Track ID</th>
                                <th>Incident Type</th>
                                <th>Zone</th>
                                <th>Severity</th>
                                <th>Conf</th>
                                <th>Action Taken</th>
                            </tr>
                        </thead>
                        <tbody id="events-tbody">
                            <!-- Filled dynamically via SSE / Polling -->
                            <tr>
                                <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 16px;">
                                    Monitoring CCTV stream for safety events...
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Right: Telemetry & Operator Controls -->
        <div>
            <!-- Real-Time Telemetry -->
            <div class="card">
                <div class="card-title">Live Measured Telemetry</div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Live Pipeline Speed:</span>
                    <span class="telemetry-value" style="color: var(--accent);" id="tel-fps">-- FPS</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">End-to-End Latency:</span>
                    <span class="telemetry-value" style="color: #34D399;" id="tel-latency">-- ms</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Detection Target (>90%):</span>
                    <span class="telemetry-value" style="color: #60A5FA;" id="tel-accuracy">100.0% (synthetic benchmark)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Train Presence State:</span>
                    <span class="telemetry-value" id="tel-train">ABSENT</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Laptop Hardware Buzzer:</span>
                    <span class="telemetry-value" id="tel-buzzer">ACTIVE</span>
                </div>
            </div>

            <!-- Hardware Buzzer Controls -->
            <div class="card">
                <div class="card-title">Hardware Buzzer Controls</div>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 0;">
                    Audible PC alarm with severity patterns (L1 pulse, L2 repeating, L3 continuous siren). Auto-stops when zone clears or on acknowledge.
                </p>
                <div style="display: flex; gap: 8px; margin-top: 12px;">
                    <button class="btn btn-secondary" id="btn-mute" onclick="toggleBuzzerMute()">Mute Buzzer</button>
                    <button class="btn btn-secondary" id="btn-test-beep" onclick="testBuzzer('L2')">Test Beep (L2)</button>
                    <button class="btn btn-secondary" id="btn-test-siren" onclick="testBuzzer('L3')">Test Siren (L3)</button>
                </div>
            </div>

            <!-- Emergency Trip Switch (ETS) -->
            <div class="card">
                <div class="card-title">Emergency Trip Switch (ETS)</div>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 0;">
                    Simulated traction power cut interlock with operator safety verification.
                </p>
                <div style="margin: 14px 0;">
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">CURRENT ETS STATE:</div>
                    <div style="font-size: 24px; font-weight: 800; color: #10B981;" id="ets-state-display">DISENGAGED</div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn btn-danger" id="btn-engage-ets" onclick="triggerETS()">ENGAGE ETS (POWER CUT)</button>
                    <button class="btn btn-secondary" id="btn-reset-ets" onclick="resetETS()">Reset</button>
                </div>
            </div>

            <!-- Simulated Train State Controller -->
            <div class="card">
                <div class="card-title">Simulate Train Movement</div>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 0;">
                    Escalation severity elevates automatically when a train is approaching.
                </p>
                <div style="display: flex; gap: 8px; margin-top: 10px;" id="train-state-panel">
                    <button class="btn btn-secondary" id="btn-train-absent" onclick="setTrainState('ABSENT')">Absent</button>
                    <button class="btn btn-secondary" id="btn-train-approaching" onclick="setTrainState('APPROACHING')">Approaching</button>
                    <button class="btn btn-secondary" id="btn-train-present" onclick="setTrainState('PRESENT')">Present</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Error Toast Notifications Container -->
    <div id="toast-container" style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 8px; pointer-events: none;"></div>

    <!-- JavaScript Live Polling & SSE Engine -->
    <script>
        let isMuted = false;
        let consecutiveFailedPolls = 0;
        let lastTelemetryTime = null;
        let isBackendDead = false;

        function setBackendDeadState(dead) {
            if (isBackendDead === dead) return;
            isBackendDead = dead;
            const banner = document.getElementById('backend-dead-banner');
            const noSignal = document.getElementById('no-signal-placeholder');
            const deadTime = document.getElementById('backend-dead-time');
            const buttons = document.querySelectorAll('.btn');

            if (dead) {
                if (banner) banner.style.display = 'block';
                if (noSignal) noSignal.style.display = 'flex';
                if (deadTime) {
                    deadTime.textContent = 'Last update: ' + (lastTelemetryTime ? lastTelemetryTime.toLocaleTimeString() : 'Never');
                }
                buttons.forEach(b => b.disabled = true);
            } else {
                if (banner) banner.style.display = 'none';
                if (noSignal) noSignal.style.display = 'none';
                buttons.forEach(b => b.disabled = false);
            }
        }

        function showErrorToast(msg) {
            const container = document.getElementById('toast-container');
            if (!container) return;
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.innerHTML = '⚠️ <strong>Action Failed:</strong> ' + msg;
            container.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transition = 'opacity 0.4s ease';
                setTimeout(() => toast.remove(), 400);
            }, 4000);
        }

        function apiPost(url, payload = {}) {
            return fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(r => {
                if (!r.ok) {
                    throw new Error(`HTTP ${r.status} ${r.statusText}`);
                }
                return r.json().catch(() => ({ status: 'ok' }));
            })
            .then(data => {
                updateTelemetry();
                return data;
            })
            .catch(err => {
                console.error('API POST error on', url, err);
                showErrorToast(`Failed to execute request to ${url}: ${err.message}`);
                throw err;
            });
        }

        function handleVideoError(img) {
            console.warn("MJPEG feed disconnected. Displaying visible stream error and retrying in 1.2s...");
            const overlay = document.getElementById('video-error-overlay');
            if (overlay) overlay.style.display = 'flex';
            setTimeout(() => {
                img.src = '/video_feed?t=' + Date.now();
            }, 1200);
        }

        function handleVideoLoad() {
            const overlay = document.getElementById('video-error-overlay');
            if (overlay) overlay.style.display = 'none';
        }

        function updateTelemetry() {
            fetch('/api/telemetry')
                .then(r => {
                    if (!r.ok) throw new Error('Telemetry HTTP ' + r.status);
                    return r.json();
                })
                .then(data => {
                    consecutiveFailedPolls = 0;
                    lastTelemetryTime = new Date();
                    setBackendDeadState(false);

                    document.getElementById('tel-fps').textContent = (data.live_fps || 0).toFixed(1) + ' FPS';
                    document.getElementById('tel-latency').textContent = (data.live_latency_ms || 0).toFixed(1) + ' ms';
                    document.getElementById('tel-accuracy').textContent = data.synthetic_accuracy || '--';
                    document.getElementById('tel-train').textContent = data.train_state || '--';
                    document.getElementById('train-state-badge').textContent = 'TRAIN: ' + (data.train_state || '--');

                    // Train Movement Panel Highlight strictly follows backend train state
                    const state = (data.train_state || 'ABSENT').toUpperCase();
                    const btnAbsent = document.getElementById('btn-train-absent');
                    const btnApp = document.getElementById('btn-train-approaching');
                    const btnPres = document.getElementById('btn-train-present');
                    if (btnAbsent && btnApp && btnPres) {
                        btnAbsent.classList.toggle('btn-active-train', state === 'ABSENT');
                        btnApp.classList.toggle('btn-active-train', state === 'APPROACHING');
                        btnPres.classList.toggle('btn-active-train', state === 'PRESENT');
                    }

                    // Camera Motion Banner
                    const cmBanner = document.getElementById('camera-motion-banner');
                    if (data.camera_motion) {
                        cmBanner.style.display = 'block';
                        document.getElementById('camera-motion-reason').textContent = data.camera_motion_reason || 'Motion detected';
                    } else {
                        cmBanner.style.display = 'none';
                    }

                    // Train state color in header badge
                    const tBadge = document.getElementById('train-state-badge');
                    if (state === 'APPROACHING') {
                        tBadge.className = 'badge badge-l2';
                    } else if (state === 'PRESENT') {
                        tBadge.className = 'badge badge-l1';
                    } else {
                        tBadge.className = 'badge badge-secure';
                    }

                    // Online & Fallback status
                    const sBadge = document.getElementById('stream-status');
                    const fbBanner = document.getElementById('fallback-banner');
                    if (data.is_fallback) {
                        fbBanner.style.display = 'block';
                        document.getElementById('fallback-target').textContent = 'Attempted source: ' + (data.source_target || 'http://192.168.1.32:8080/video');
                        document.getElementById('fallback-retry').textContent = data.retry_status || 'Retrying background connect...';
                        sBadge.textContent = 'DEMO FALLBACK';
                        sBadge.className = 'badge badge-l2';
                    } else {
                        fbBanner.style.display = 'none';
                        if (data.is_online) {
                            sBadge.textContent = 'LIVE STREAM';
                            sBadge.className = 'badge badge-live';
                        } else {
                            sBadge.textContent = 'RECONNECTING...';
                            sBadge.className = 'badge badge-l2';
                        }
                    }

                    // Buzzer
                    const bVal = document.getElementById('tel-buzzer');
                    if (data.buzzer) {
                        isMuted = data.buzzer.muted;
                        document.getElementById('btn-mute').textContent = isMuted ? 'Unmute Buzzer' : 'Mute Buzzer';
                        if (isMuted) {
                            bVal.textContent = 'MUTED';
                            bVal.style.color = '#F59E0B';
                        } else if (data.buzzer.active) {
                            bVal.textContent = 'BUZZING (' + data.buzzer.pattern + ')';
                            bVal.style.color = '#EF4444';
                        } else {
                            bVal.textContent = 'STANDBY';
                            bVal.style.color = '#10B981';
                        }
                    }

                    // ETS
                    const etsDisp = document.getElementById('ets-state-display');
                    const etsBadge = document.getElementById('ets-badge');
                    etsDisp.textContent = data.ets_status;
                    if (data.ets_status === 'ENGAGED') {
                        etsDisp.style.color = '#EF4444';
                        etsBadge.textContent = 'POWER CUT (ENGAGED)';
                        etsBadge.className = 'badge badge-l3';
                    } else {
                        etsDisp.style.color = '#10B981';
                        etsBadge.textContent = 'ETS READY';
                        etsBadge.className = 'badge badge-l1';
                    }

                    // Overlooker Alert Banner
                    const banner = document.getElementById('alert-banner');
                    if (data.active_alert) {
                        banner.style.display = 'block';
                        document.getElementById('alert-title').textContent = '🚨 ' + data.active_alert.title;
                        document.getElementById('alert-body').textContent = data.active_alert.body.replace('\\n', ' | ');
                    } else {
                        banner.style.display = 'none';
                    }

                    // Adaptive Zones Unverified Banner
                    const zoneBanner = document.getElementById('zone-unverified-banner');
                    const zoneBtn = document.getElementById('btn-confirm-zones');
                    const az = data.adaptive_zones;
                    if (zoneBanner && az && (az.zones_unverified || az.requires_operator_confirmation || az.status === 'UNVERIFIED_REFERENCE' || az.status === 'ZONES UNVERIFIED')) {
                        zoneBanner.style.display = 'block';
                        const isUnverifiedRef = (az.status === 'UNVERIFIED_REFERENCE');
                        document.getElementById('zone-alert-title').textContent = isUnverifiedRef
                            ? '⚠️ UNVERIFIED REFERENCE FRAME — OPERATOR CONFIRMATION REQUIRED'
                            : '⚠️ ZONES UNVERIFIED — CAMERA DRIFT DETECTED';
                        document.getElementById('zone-alert-body').textContent = `Drift: ${(az.drift_px||0).toFixed(1)}px | Inliers: ${az.inliers||0} | Status: ${az.status}${az.failure_reason ? ' (' + az.failure_reason + ')' : ''}`;
                        if (zoneBtn && !isBackendDead) zoneBtn.disabled = false;
                    } else if (zoneBanner) {
                        zoneBanner.style.display = 'none';
                    }
                })
                .catch(err => {
                    console.debug('Telemetry poll error:', err);
                    consecutiveFailedPolls++;
                    if (consecutiveFailedPolls >= 3) {
                        setBackendDeadState(true);
                    }
                });
        }

        let currentEventFilter = 'all';
        function setEventFilter(f) {
            currentEventFilter = f;
            document.querySelectorAll('#event-filter-group .filter-btn').forEach(b => b.classList.remove('active'));
            const target = document.getElementById('filter-' + f);
            if (target) target.classList.add('active');
            updateEvents();
        }

        function updateEvents() {
            fetch('/api/events')
                .then(r => {
                    if (!r.ok) throw new Error('Events HTTP ' + r.status);
                    return r.json();
                })
                .then(data => {
                    const tbody = document.getElementById('events-tbody');
                    const allEvents = data.events || [];

                    const events = allEvents.filter(ev => {
                        const isAnom = ev.is_anomaly || ['PERSON_DOWN', 'LOITERING', 'ABANDONED_OBJECT', 'POSSIBLE_SNATCH', 'ANOMALY_POSE'].some(t => (ev.event_type || '').indexOf(t) !== -1);
                        if (currentEventFilter === 'safety') return !isAnom;
                        if (currentEventFilter === 'anomaly') return isAnom;
                        return true;
                    });

                    document.getElementById('event-count').textContent = events.length + ' events' + (currentEventFilter !== 'all' ? ` (${currentEventFilter})` : '');

                    if (events.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 16px;">No ${currentEventFilter} events recorded.</td></tr>`;
                        return;
                    }

                    let html = '';
                    events.slice(0, 20).forEach(ev => {
                        let sevBadge = 'badge-secure';
                        if (ev.severity === 'L3') sevBadge = 'badge-l3';
                        else if (ev.severity === 'L2') sevBadge = 'badge-l2';
                        else if (ev.severity === 'L1') sevBadge = 'badge-l1';
                        else if (ev.severity === 'DISTRESS') sevBadge = 'badge-distress';

                        const timeStr = ev.time_str || (ev.timestamp ? new Date(ev.timestamp * 1000).toLocaleTimeString() : '--');
                        const actionStr = ev.action || '--';
                        const conf = ev.confidence ? (ev.confidence * 100).toFixed(0) + '%' : '--';

                        const isDemo = ev.is_demo || (ev.event_type && ev.event_type.indexOf('[DEMO]') !== -1);
                        const cleanType = ev.event_type ? ev.event_type.replace('[DEMO]', '').trim() : '--';
                        const demoBadge = isDemo ? '<span class="badge badge-l2" style="font-size: 10px; margin-right: 4px;">DEMO</span>' : '';

                        const isAnom = ev.is_anomaly || ['PERSON_DOWN', 'LOITERING', 'ABANDONED_OBJECT', 'POSSIBLE_SNATCH', 'ANOMALY_POSE'].some(t => cleanType.indexOf(t) !== -1);
                        const anomBadge = isAnom ? '<span class="badge badge-anomaly" style="font-size: 10px; margin-right: 4px;">ANOMALY</span>' : '';

                        const snapThumb = ev.snapshot_crop ? `<img src="${ev.snapshot_crop}" style="width:24px;height:24px;object-fit:cover;border-radius:3px;vertical-align:middle;margin-right:4px;" title="Object crop"/>` : '';

                        html += `<tr>
                            <td>${timeStr}</td>
                            <td>#${ev.track_id}</td>
                            <td>${demoBadge}${anomBadge}${snapThumb}<strong>${cleanType}</strong></td>
                            <td>${ev.zone}</td>
                            <td><span class="badge ${sevBadge}">${ev.severity}</span></td>
                            <td>${conf}</td>
                            <td>${actionStr}</td>
                        </tr>`;
                    });
                    tbody.innerHTML = html;
                })
                .catch(err => console.debug('Events poll error:', err));
        }

        // Action Handlers
        function acknowledgeAlarm() {
            apiPost('/api/acknowledge').then(() => {
                const b = document.getElementById('alert-banner');
                if (b) b.style.display = 'none';
            });
        }

        function toggleBuzzerMute() {
            apiPost('/api/buzzer/toggle_mute');
        }

        function testBuzzer(pattern) {
            apiPost('/api/buzzer/test', { pattern: pattern });
        }

        function triggerETS() {
            if (confirm("Confirm Manual Traction Power Cut (Emergency Trip Switch)?")) {
                apiPost('/api/ets/trigger');
            }
        }

        function resetETS() {
            apiPost('/api/ets/reset');
        }

        function confirmZoneAlignment() {
            const btn = document.getElementById('btn-confirm-zones');
            if (btn) btn.disabled = true;
            apiPost('/api/zones/confirm')
                .then(res => {
                    console.log('Zone alignment confirmed:', res);
                    updateTelemetry();
                })
                .catch(err => {
                    showErrorToast('Zone Confirmation Failed: ' + (err.message || err));
                    if (btn && !isBackendDead) btn.disabled = false;
                });
        }

        function setTrainState(state) {
            apiPost('/api/train_state', { state: state });
        }

        // Live Poll Loops (Fast for telemetry, regular for events)
        setInterval(updateTelemetry, 400);
        setInterval(updateEvents, 1000);
        updateTelemetry();
        updateEvents();

        // Optional SSE connection for instantaneous event push
        if (!!window.EventSource) {
            const evtSource = new EventSource("/api/events/stream");
            evtSource.onmessage = function(e) {
                updateEvents();
                updateTelemetry();
            };
        }
    </script>
</body>
</html>
    """

    @app.route("/")
    def dashboard():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/video_feed")
    def video_feed():
        """MJPEG video stream endpoint for dashboard <img>."""
        resp = Response(
            active_hub.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

    @app.route("/video_frame.jpg")
    @app.route("/api/snapshot")
    def video_frame():
        """Returns the latest single JPEG snapshot frame."""
        jpeg_bytes = active_hub.get_latest_jpeg()
        resp = Response(jpeg_bytes, mimetype="image/jpeg")
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        return resp

    @app.route("/api/telemetry")
    def api_telemetry():
        return jsonify(active_hub.get_telemetry())

    @app.route("/api/events")
    def api_events():
        events = active_hub.get_events(limit=50)
        return jsonify({
            "events": events,
            "telemetry": active_hub.get_telemetry()
        })

    @app.route("/api/events/stream")
    def api_events_stream():
        """Server-Sent Events endpoint pushing events immediately as they occur."""
        q = active_hub.add_sse_listener()

        def stream():
            try:
                # Send initial ping
                yield "data: {\"type\": \"init\"}\n\n"
                while True:
                    try:
                        ev = q.get(timeout=25.0)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except Exception:
                        # Heartbeat comment to keep HTTP connection alive
                        yield ": keep-alive\n\n"
            finally:
                active_hub.remove_sse_listener(q)

        return Response(stream(), mimetype="text/event-stream")

    @app.route("/api/acknowledge", methods=["POST"])
    def api_acknowledge():
        return jsonify(active_hub.acknowledge_alarm())

    @app.route("/api/buzzer/toggle_mute", methods=["POST"])
    def api_buzzer_toggle_mute():
        return jsonify(active_hub.toggle_buzzer_mute())

    @app.route("/api/buzzer/test", methods=["POST"])
    def api_buzzer_test():
        data = request.get_json(silent=True) or {}
        pattern = data.get("pattern", "L2")
        return jsonify(active_hub.test_buzzer(pattern=pattern))

    @app.route("/api/ets/trigger", methods=["POST"])
    def api_ets_trigger():
        return jsonify(active_hub.trip_ets())

    @app.route("/api/ets/reset", methods=["POST"])
    def api_ets_reset():
        return jsonify(active_hub.reset_ets())

    @app.route("/api/train_state", methods=["POST"])
    def api_train_state():
        data = request.get_json(silent=True) or {}
        new_state = data.get("state", "ABSENT")
        return jsonify(active_hub.set_train_state(new_state))

    @app.route("/api/alert", methods=["POST"])
    def api_external_alert():
        """Webhook endpoint for external systems."""
        data = request.get_json(silent=True) or {}
        active_hub.push_event(data)
        return jsonify({"status": "received"})

    @app.route("/api/zones/confirm", methods=["POST"])
    def api_zones_confirm():
        """Operator confirmation of current or proposed zone alignment."""
        expected_key = os.environ.get("API_KEY")
        if expected_key:
            req_key = request.headers.get("X-API-Key") or request.args.get("key")
            if req_key != expected_key:
                return jsonify({"status": "error", "message": "Unauthorized"}), 403
        res = active_hub.confirm_zone_alignment()
        status_code = 200 if res.get("status") == "ok" else 400
        return jsonify(res), status_code

    return app

# Default app instance
app = create_dashboard_app(get_global_hub())

if __name__ == "__main__":
    print("[Dashboard] Running standalone on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

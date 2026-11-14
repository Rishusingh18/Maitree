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
        .badge-secure { background-color: rgba(16, 185, 129, 0.2); color: #A7F3D0; border: 1px solid var(--success); }

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

    <!-- Main Grid -->
    <div class="grid">
        <!-- Left: Live Video & Event Log -->
        <div>
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
                <div class="video-container">
                    <img id="live-video" src="/video_feed" alt="CCTV Stream" onerror="reconnectVideo(this);">
                    <div class="video-overlay-badge">
                        <span class="badge badge-l3" id="ets-badge">ETS READY</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">
                    <span>Recent Incident Event Log</span>
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
                    <button class="btn btn-secondary" onclick="testBuzzer('L2')">Test Beep (L2)</button>
                    <button class="btn btn-secondary" onclick="testBuzzer('L3')">Test Siren (L3)</button>
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
                    <button class="btn btn-danger" onclick="triggerETS()">ENGAGE ETS (POWER CUT)</button>
                    <button class="btn btn-secondary" onclick="resetETS()">Reset</button>
                </div>
            </div>

            <!-- Simulated Train State Controller -->
            <div class="card">
                <div class="card-title">Simulate Train Movement</div>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 0;">
                    Escalation severity elevates automatically when a train is approaching.
                </p>
                <div style="display: flex; gap: 8px; margin-top: 10px;">
                    <button class="btn btn-secondary" onclick="setTrainState('ABSENT')">Absent</button>
                    <button class="btn btn-warning" onclick="setTrainState('APPROACHING')">Approaching</button>
                    <button class="btn btn-secondary" onclick="setTrainState('PRESENT')">Present</button>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript Live Polling & SSE Engine -->
    <script>
        let isMuted = false;

        function reconnectVideo(img) {
            console.warn("MJPEG feed disconnected. Reconnecting in 1.2s...");
            setTimeout(() => {
                img.src = '/video_feed?t=' + Date.now();
            }, 1200);
        }

        function updateTelemetry() {
            fetch('/api/telemetry')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('tel-fps').textContent = data.live_fps.toFixed(1) + ' FPS';
                    document.getElementById('tel-latency').textContent = data.live_latency_ms.toFixed(1) + ' ms';
                    document.getElementById('tel-accuracy').textContent = data.synthetic_accuracy;
                    document.getElementById('tel-train').textContent = data.train_state;
                    document.getElementById('train-state-badge').textContent = 'TRAIN: ' + data.train_state;

                    // Train state color
                    const tBadge = document.getElementById('train-state-badge');
                    if (data.train_state === 'APPROACHING') {
                        tBadge.className = 'badge badge-l2';
                    } else if (data.train_state === 'PRESENT') {
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
                })
                .catch(err => console.debug('Telemetry poll error:', err));
        }

        function updateEvents() {
            fetch('/api/events')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('events-tbody');
                    const events = data.events || [];
                    document.getElementById('event-count').textContent = events.length + ' events';

                    if (events.length === 0) return;

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

                        html += `<tr>
                            <td>${timeStr}</td>
                            <td>#${ev.track_id}</td>
                            <td>${demoBadge}<strong>${cleanType}</strong></td>
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
            fetch('/api/acknowledge', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    document.getElementById('alert-banner').style.display = 'none';
                    updateTelemetry();
                });
        }

        function toggleBuzzerMute() {
            fetch('/api/buzzer/toggle_mute', { method: 'POST' })
                .then(r => r.json())
                .then(() => updateTelemetry());
        }

        function testBuzzer(pattern) {
            fetch('/api/buzzer/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pattern: pattern })
            }).then(() => updateTelemetry());
        }

        function triggerETS() {
            if (confirm("Confirm Manual Traction Power Cut (Emergency Trip Switch)?")) {
                fetch('/api/ets/trigger', { method: 'POST' }).then(() => updateTelemetry());
            }
        }

        function resetETS() {
            fetch('/api/ets/reset', { method: 'POST' }).then(() => updateTelemetry());
        }

        function setTrainState(state) {
            fetch('/api/train_state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: state })
            }).then(() => updateTelemetry());
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

    return app

# Default app instance
app = create_dashboard_app(get_global_hub())

if __name__ == "__main__":
    print("[Dashboard] Running standalone on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

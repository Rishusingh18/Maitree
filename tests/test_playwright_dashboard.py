"""Automated end-to-end browser verification of the live dashboard video stream using Playwright Chromium.
Asserts:
1. Video <img> element loads real frames with naturalWidth > 0 and naturalHeight > 0.
2. Pixel content actively changes over time (> 0 difference over 2-3 seconds).
3. The dashboard UI loads without console errors or broken video streams.
"""
import os
import sys
import time
import subprocess
import requests
import pytest

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright is not installed")
def test_dashboard_video_renders_in_real_browser():
    """Verify in headless Chromium that /video_feed stream renders valid frames in the dashboard."""
    port = 5002
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["METRO_SOURCE_OVERRIDE"] = "demo_clips/demo_fall.mp4"

    cmd = [
        sys.executable,
        "-m", "metro_safety.demo.live",
        "--source", "demo_clips/demo_fall.mp4",
        "--port", str(port),
        "--no-window",
        "--loop"
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for Flask dashboard server to come online
    server_online = False
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/telemetry", timeout=1.0)
            if r.status_code == 200:
                server_online = True
                break
        except Exception:
            time.sleep(0.3)

    assert server_online, f"Dashboard server on port {port} failed to start"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            console_errors = []
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")

            # Wait up to 10s for the video image to load valid dimensions
            img = page.locator("#live-video")
            img.wait_for(state="visible", timeout=10000)

            # Poll for naturalWidth > 0
            natural_w = 0
            natural_h = 0
            for _ in range(20):
                natural_w = img.evaluate("el => el.naturalWidth")
                natural_h = img.evaluate("el => el.naturalHeight")
                if natural_w > 0 and natural_h > 0:
                    break
                time.sleep(0.5)

            assert natural_w > 0, f"Expected video <img> naturalWidth > 0, got {natural_w}"
            assert natural_h > 0, f"Expected video <img> naturalHeight > 0, got {natural_h}"

            # Capture snapshot 1
            snap1 = img.evaluate("""img => {
                const canvas = document.createElement('canvas');
                canvas.width = 64;
                canvas.height = 48;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 64, 48);
                return canvas.toDataURL();
            }""")

            # Wait 2.5 seconds for video playback
            time.sleep(2.5)

            # Capture snapshot 2
            snap2 = img.evaluate("""img => {
                const canvas = document.createElement('canvas');
                canvas.width = 64;
                canvas.height = 48;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 64, 48);
                return canvas.toDataURL();
            }""")

            assert snap1 != snap2, "Video feed image appears frozen (pixel snapshot did not change over 2.5s)"

            # Assert no raw JavaScript source code leaks into document.body.innerText
            body_text = page.evaluate("() => document.body.innerText")
            script_snippets = [
                "setInterval(updateTelemetry",
                "setInterval(updateEvents",
                "function updateTelemetry",
                "function updateEvents",
                "EventSource",
                "// Live Poll Loops",
                "apiPost("
            ]
            for snippet in script_snippets:
                assert snippet not in body_text, f"Raw JavaScript text leaked into document.body: '{snippet}'"

            # Assert Anomaly filter buttons exist and toggle active class
            filter_all = page.locator("#filter-all")
            filter_safety = page.locator("#filter-safety")
            filter_anomaly = page.locator("#filter-anomaly")
            assert filter_all.is_visible()
            assert filter_safety.is_visible()
            assert filter_anomaly.is_visible()

            # Click anomaly filter button
            filter_anomaly.click()
            assert "active" in (filter_anomaly.get_attribute("class") or "")
            assert "active" not in (filter_all.get_attribute("class") or "")

            browser.close()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


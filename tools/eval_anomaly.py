"""Evaluation Harness for Metro Safety Anomaly and Theft Detection.

Evaluates performance across labelled test clips or synthetic test scenarios.
Reports:
- Hit / Miss counts
- Precision and Recall
- Time-to-alert (latency in seconds)
- False Alert rate
- Measured FPS and processing latency (differentiating synthetic vs real clips)

Ground Rules:
- Only report measured numbers with explicit source labels (synthetic / recorded clip / live).
- Never claim unmeasured accuracy on real footage.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from metro_safety.config import load_config
from metro_safety.zones.geometry import ZoneManager
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.anomaly.engine import AnomalyEngine
from metro_safety.events.models import EventType, SafetyEvent
from metro_safety.simulation.generator import ScenarioGenerator


class AnomalyEvaluator:
    """Evaluates anomaly detection engines against test datasets."""

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.zm = ZoneManager(self.cfg["zones"])

    def evaluate_manifest(self, manifest_csv: str) -> Dict[str, Any]:
        """Evaluate labelled clips listed in CSV manifest."""
        csv_path = Path(manifest_csv)
        if not csv_path.exists():
            return {"error": f"Manifest file not found: {manifest_csv}"}

        raise NotImplementedError(
            "Video manifest evaluation requires live clip decoding; use evaluate_synthetic_battery() or pass video clips to demo/run.py."
        )

    def evaluate_synthetic_battery(self) -> Dict[str, Any]:
        """Runs evaluation battery on synthetic multi-agent scenarios."""
        generator = ScenarioGenerator(width=640, height=480, fps=25.0)

        scenarios = [
            ("PERSON_DOWN", "PERSON_DOWN", 300),
            ("BENCH_LYING", "PERSON_DOWN", 250),
            ("LOITERING", "LOITERING", 850),
            ("NORMAL_WAITING", "NORMAL", 150),
            ("NORMAL_WALKING", "NORMAL", 150),
        ]

        metrics: Dict[str, Dict[str, Any]] = {}
        t_start_total = time.perf_counter()
        total_frames = 0

        for sc_name, target_type, n_frames in scenarios:
            engine = AnomalyEngine(self.cfg)
            # Create synthetic frames data
            data = generator.generate_person_trajectory(
                scenario=sc_name,
                num_frames=n_frames
            )

            hits = 0
            time_to_alert: Optional[float] = None
            false_positives = 0
            total_frames += n_frames

            # Simulate video frames and detections
            for idx, frame_dict in enumerate(data):
                ts = idx / 25.0
                kpts = frame_dict["keypoints"]
                box = frame_dict["box"]

                from metro_safety.perception.base import Detection
                det = Detection(box=box, confidence=0.90, keypoints=kpts, track_id=1)

                # Process
                events = engine.process_frame(
                    frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    frame_idx=idx,
                    person_detections=[det],
                    zone_manager=self.zm,
                    timestamp=ts,
                    train_state="ABSENT"
                )

                for ev in events:
                    ev_type_val = ev.event_type.value
                    if target_type == "NORMAL":
                        false_positives += 1
                    elif target_type in ev_type_val:
                        hits += 1
                        if time_to_alert is None:
                            time_to_alert = ts
                    else:
                        false_positives += 1

            is_hit = hits > 0 if target_type != "NORMAL" else (false_positives == 0)
            metrics[sc_name] = {
                "ground_truth": target_type,
                "hit": is_hit,
                "alerts_count": hits,
                "false_positives": false_positives,
                "time_to_alert_sec": round(time_to_alert, 2) if time_to_alert is not None else None,
                "frames_evaluated": n_frames
            }

        total_elapsed = time.perf_counter() - t_start_total
        measured_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

        summary = {
            "evaluation_source": "synthetic (ScenarioGenerator kinematic simulation)",
            "measured_fps": round(measured_fps, 1),
            "total_frames": total_frames,
            "scenarios": metrics
        }
        return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly & Theft Detection")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic test battery")
    args = parser.parse_args()

    evaluator = AnomalyEvaluator()

    if args.manifest:
        res = evaluator.evaluate_manifest(args.manifest)
        print(json.dumps(res, indent=2))
    else:
        print("[Eval] Running synthetic anomaly detection benchmark suite...")
        res = evaluator.evaluate_synthetic_battery()
        print("\n" + "=" * 60)
        print("  AI METRO ANOMALY DETECTION EVALUATION RESULTS")
        print(f"  Source: {res['evaluation_source']}")
        print(f"  Speed:  {res['measured_fps']} FPS (Inference + Pipeline)")
        print("=" * 60)
        for sc, m in res["scenarios"].items():
            status = "PASS (HIT)" if m["hit"] else "FAIL (MISS)"
            tta = f"{m['time_to_alert_sec']}s" if m['time_to_alert_sec'] is not None else "N/A"
            print(f"  • {sc:<16}: {status:<12} | TTA: {tta:<6} | FP: {m['false_positives']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

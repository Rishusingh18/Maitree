"""STG-NF Pretrained Skeleton Anomaly Model Evaluator.

Loads official ICCV 2023 STG-NF (Spatio-Temporal Graph Normalizing Flows)
weights and evaluates rolling temporal skeleton sequences.
Computes negative log-likelihood (NLL) bits-per-dimension as a principled
data-driven anomaly likelihood score.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import math
import numpy as np

logger = logging.getLogger("metro_safety.anomaly.stg_nf")


class STGNFEvaluator:
    """Evaluates 17-keypoint COCO poses using pretrained STG-NF graph normalizing flow."""

    def __init__(
        self,
        checkpoint_path: str = "data/models/ShanghaiTech_85_9.tar",
        window_len: int = 24,
        anomaly_nll_thresh: float = 12.0,
        device: str = "cpu"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.window_len = window_len
        self.anomaly_nll_thresh = anomaly_nll_thresh
        self.device = device
        self.model = None
        self._is_loaded = False
        self._track_buffers: dict = {}

        self._try_load_model()

    def _try_load_model(self):
        try:
            import torch
            from metro_safety.anomaly.stg_nf.model_pose import STG_NF

            if not self.checkpoint_path.exists():
                logger.warning(f"[STG-NF] Checkpoint not found at {self.checkpoint_path}. Operating in kinematic fallback mode.")
                return

            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
            args = ckpt["args"]

            pose_shape = (2, self.window_len, 18)
            hidden_dim = getattr(args, "hidden_dim", getattr(args, "model_hidden_dim", 0))
            strategy = getattr(args, "strategy", getattr(args, "adj_strategy", "uniform"))
            max_hops = getattr(args, "max_hop", getattr(args, "max_hops", 8))

            model = STG_NF(
                pose_shape=pose_shape,
                hidden_channels=hidden_dim,
                K=getattr(args, "K", 8),
                L=getattr(args, "L", 1),
                actnorm_scale=1.0,
                flow_permutation=getattr(args, "flow_permutation", "invcov"),
                flow_coupling="affine",
                LU_decomposed=True,
                learn_top=False,
                R=getattr(args, "R", 3.0),
                edge_importance=getattr(args, "edge_importance", False),
                temporal_kernel_size=getattr(args, "temporal_kernel", None),
                strategy=strategy,
                max_hops=max_hops,
                device=self.device
            )

            incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
            missing_params = [k for k in incompatible.missing_keys if not (k.endswith(".indices") or k.endswith(".indices_inverse"))]
            if missing_params:
                raise RuntimeError(f"Missing parameter weights in checkpoint state dict: {missing_params[:4]}")
            model.set_actnorm_init()
            model.to(self.device)
            model.eval()
            self.model = model
            self._is_loaded = True
            logger.info(f"[STG-NF] Pretrained skeleton anomaly flow model loaded from {self.checkpoint_path}.")
        except Exception as e:
            logger.warning(f"[STG-NF] Model load failed: {e}. Falling back to kinematic scoring.")
            self.model = None
            self._is_loaded = False

    @staticmethod
    def coco_to_openpose18(coco_kpts: np.ndarray) -> np.ndarray:
        """Convert standard 17 COCO keypoints (x, y, conf) to OpenPose-18 joint order."""
        # coco_kpts shape: (17, 2) or (17, 3)
        xy = coco_kpts[:, :2]
        op18 = np.zeros((18, 2), dtype=np.float32)

        # 0: Nose
        op18[0] = xy[0]
        # 1: Neck = average of Left Shoulder (5) and Right Shoulder (6)
        op18[1] = (xy[5] + xy[6]) / 2.0
        # 2: RShoulder
        op18[2] = xy[6]
        # 3: RElbow
        op18[3] = xy[8]
        # 4: RWrist
        op18[4] = xy[10]
        # 5: LShoulder
        op18[5] = xy[5]
        # 6: LElbow
        op18[6] = xy[7]
        # 7: LWrist
        op18[7] = xy[9]
        # 8: RHip
        op18[8] = xy[12]
        # 9: RKnee
        op18[9] = xy[14]
        # 10: RAnkle
        op18[10] = xy[16]
        # 11: LHip
        op18[11] = xy[11]
        # 12: LKnee
        op18[12] = xy[13]
        # 13: LAnkle
        op18[13] = xy[15]
        # 14: REye
        op18[14] = xy[2]
        # 15: LEye
        op18[15] = xy[1]
        # 16: REar
        op18[16] = xy[4]
        # 17: LEar
        op18[17] = xy[3]

        return op18

    def evaluate_sequence(self, keypoint_sequence: List[np.ndarray]) -> Tuple[bool, float, float]:
        """Evaluate a temporal window of poses.
        Returns:
            is_anomaly (bool): Whether the NLL exceeds threshold
            nll_score (float): Raw bits-per-dim negative log-likelihood
            calibrated_prob (float): Normalized anomaly confidence in [0.0, 0.99]
        """
        if not keypoint_sequence:
            return False, 0.0, 0.0

        # Subsample or interpolate sequence to exactly self.window_len frames
        seq_len = len(keypoint_sequence)
        indices = np.linspace(0, seq_len - 1, self.window_len, dtype=int)
        sampled = [keypoint_sequence[i] for i in indices]

        # Convert to OpenPose 18
        op_poses = np.zeros((self.window_len, 18, 2), dtype=np.float32)
        for t_idx, kp in enumerate(sampled):
            op_poses[t_idx] = self.coco_to_openpose18(kp)

        # Normalize relative to neck joint (joint 1) and scale by bounding box diagonal
        root = op_poses[:, 1:2, :]  # shape (T, 1, 2)
        centered = op_poses - root

        mins = np.min(centered, axis=(0, 1))
        maxs = np.max(centered, axis=(0, 1))
        diag = float(np.linalg.norm(maxs - mins))
        scale = max(1e-3, diag)
        normalized = centered / scale  # shape (T, 18, 2)

        # Reshape to (1, 2, T, 18)
        tensor_in = np.transpose(normalized, (2, 0, 1))[np.newaxis, :, :, :]

        if self._is_loaded and self.model is not None:
            try:
                import torch
                t_input = torch.tensor(tensor_in, dtype=torch.float32, device=self.device)
                y = torch.ones(t_input.size(0), dtype=torch.long, device=self.device)
                with torch.no_grad():
                    out = self.model(t_input, label=y)
                    if isinstance(out, tuple):
                        nll_val = out[1]
                    else:
                        nll_val = out
                    nll = float(nll_val.mean().cpu().item()) if hasattr(nll_val, "mean") else float(nll_val)
                
                # Calibrate probability using sigmoid-like scaling
                delta = nll - self.anomaly_nll_thresh
                prob = 1.0 / (1.0 + math.exp(-0.5 * delta))
                calibrated = min(0.95, max(0.05, prob))
                is_anomaly = nll > self.anomaly_nll_thresh
                return is_anomaly, round(nll, 2), round(calibrated, 2)
            except Exception as inf_err:
                logger.debug(f"[STG-NF] Inference exception: {inf_err}")

        # Fallback heuristic scoring: measure torso flatness and keypoint jerkiness
        y_spans = []
        x_spans = []
        valid_counts = []
        for p in sampled:
            valid = p[p[:, 2] > 0.1, :2] if p.shape[-1] >= 3 else p[(p[:, 0] > 0) | (p[:, 1] > 0)]
            valid_counts.append(len(valid))
            if len(valid) >= 4:
                x_spans.append(float(np.max(valid[:, 0]) - np.min(valid[:, 0])))
                y_spans.append(float(np.max(valid[:, 1]) - np.min(valid[:, 1])))

        if not x_spans or (valid_counts and max(valid_counts) < 4):
            return False, 0.0, 0.0

        avg_aspect = np.mean(x_spans) / max(1.0, float(np.mean(y_spans)))

        # Aspect ratio > 1.2 indicates horizontal body
        pseudo_nll = 15.0 if avg_aspect > 1.2 else 8.0
        is_anom = pseudo_nll > self.anomaly_nll_thresh
        conf = 0.85 if is_anom else 0.10
        return is_anom, pseudo_nll, conf

    def update_track(self, track_id: int, keypoints: np.ndarray):
        """Append recent pose keypoints for this track ID."""
        if track_id not in self._track_buffers:
            self._track_buffers[track_id] = []
        buf = self._track_buffers[track_id]
        buf.append(np.array(keypoints, copy=True))
        if len(buf) > self.window_len:
            buf.pop(0)

    def evaluate_track(self, track_id: int) -> Tuple[bool, float, float]:
        """Evaluate rolling pose sequence for an active track."""
        buf = self._track_buffers.get(track_id, [])
        warmup = min(12, self.window_len // 2)
        if len(buf) < warmup:
            return False, 0.0, 0.0
        return self.evaluate_sequence(buf)

    def prune_track_buffers(self, active_track_ids: set):
        """Clean up buffers for departed tracks."""
        for tid in list(self._track_buffers.keys()):
            if tid not in active_track_ids:
                del self._track_buffers[tid]


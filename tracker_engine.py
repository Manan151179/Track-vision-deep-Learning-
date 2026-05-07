"""
tracker_engine.py — Backend processing engine for the MOT16 Multi-Object Tracker.

Extracts the inference pipeline from the Jupyter notebook into a clean,
importable module.  The Streamlit app (app.py) calls `process_video()` which:

  1. Loads the fine-tuned Faster R-CNN detector and Siamese ReID model.
  2. Reads an input video frame-by-frame.
  3. Runs detection → embedding → cosine-similarity tracking → annotated output.
  4. Writes the annotated video and returns statistics.
"""

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image
from typing import Callable, Optional
from filterpy.kalman import KalmanFilter


# ---------------------------------------------------------------------------
# Siamese Network (Re-ID model — identical to training definition)
# ---------------------------------------------------------------------------
class Siamese_Network(nn.Module):
    """
    Custom 3-layer CNN → 256-dim L2-normalised embedding.

    Architecture (must match the saved weights exactly):
      cnn: Conv2d(3→32) → BN → ReLU → MaxPool
           Conv2d(32→64) → BN → ReLU → MaxPool
           Conv2d(64→128) → BN → ReLU → MaxPool
      fc:  Flatten → Linear(16384→512) → ReLU → Dropout → Linear(512→256)

    Input: (B, 3, 128, 64) crops.
    After 3× (conv3×3 pad1 + pool2): spatial 128×64 → 16×8, channels 128
    ⇒ flatten dim = 128 × 16 × 8 = 16 384.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # 0
            nn.BatchNorm2d(32),                             # 1
            nn.ReLU(inplace=True),                          # 2
            nn.MaxPool2d(2),                                # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 4
            nn.BatchNorm2d(64),                             # 5
            nn.ReLU(inplace=True),                          # 6
            nn.MaxPool2d(2),                                # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8
            nn.BatchNorm2d(128),                            # 9
            nn.ReLU(inplace=True),                          # 10
            nn.MaxPool2d(2),                                # 11
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                                   # 0
            nn.Linear(128 * 16 * 8, 512),                  # 1
            nn.ReLU(inplace=True),                          # 2
            nn.Dropout(0.3),                                # 3
            nn.Linear(512, embed_dim),                      # 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, dim=1)


# ---------------------------------------------------------------------------
# Kalman Filter helpers  (DeepSORT constant-velocity model)
# ---------------------------------------------------------------------------
# State vector  : [cx, cy, a, h,  vcx, vcy, va, vh]   (8-dim)
# Measurement   : [cx, cy, a, h]                       (4-dim)
#   cx, cy = bounding-box centre;  a = aspect ratio (w/h);  h = height
#
# Using filterpy.kalman.KalmanFilter directly — no extra dependencies.

def _box_to_z(box: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] → measurement vector [cx, cy, a, h]."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    a  = w / h if h > 0 else 1.0
    return np.array([cx, cy, a, h], dtype=float)


def _z_to_box(z: np.ndarray) -> np.ndarray:
    """Convert KF state/mean [cx, cy, a, h, ...] back to [x1,y1,x2,y2]."""
    cx, cy, a, h = z[:4]
    w  = a * h
    x1 = cx - w / 2
    y1 = cy - h / 2
    return np.array([x1, y1, x1 + w, y1 + h], dtype=float)


def _make_kf(box: np.ndarray) -> KalmanFilter:
    """
    Build and initialise a constant-velocity Kalman Filter for one track.

    State transition (F): position += dt*velocity  (dt=1 frame)
    Measurement matrix (H): observe [cx, cy, a, h] only (no velocity).
    Noise matrices chosen to match the original DeepSORT paper values.
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # State transition matrix  F  (constant velocity)
    kf.F = np.eye(8, dtype=float)
    for i in range(4):
        kf.F[i, i + 4] = 1.0          # pos[i] += vel[i]

    # Measurement matrix  H  (observe first 4 components only)
    kf.H = np.zeros((4, 8), dtype=float)
    kf.H[:4, :4] = np.eye(4)

    # Measurement noise  R  — how much we trust the detector
    kf.R *= 10.0
    kf.R[2:, 2:] *= 10.0             # aspect ratio & height noisier

    # Initial state covariance  P
    kf.P[4:, 4:] *= 1000.0           # high uncertainty in initial velocity
    kf.P       *= 10.0

    # Process noise  Q  — model uncertainty per frame
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    # Initialise state from the first bounding box
    kf.x[:4] = _box_to_z(box).reshape(4, 1)
    return kf


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between every pair (a_i, b_j).
    boxes_a : (M, 4)  [x1,y1,x2,y2]
    boxes_b : (N, 4)  [x1,y1,x2,y2]
    Returns : (M, N) IoU matrix
    """
    M, N = len(boxes_a), len(boxes_b)
    iou = np.zeros((M, N), dtype=float)
    for i, a in enumerate(boxes_a):
        xx1 = np.maximum(a[0], boxes_b[:, 0])
        yy1 = np.maximum(a[1], boxes_b[:, 1])
        xx2 = np.minimum(a[2], boxes_b[:, 2])
        yy2 = np.minimum(a[3], boxes_b[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a + area_b - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Track — per-identity state object  (now contains a Kalman Filter)
# ---------------------------------------------------------------------------
class Track:
    """
    Holds the state for a single tracked identity.

    Attributes
    ----------
    track_id          : Unique integer ID assigned at track creation.
    kf                : filterpy KalmanFilter for motion prediction/update.
                        State: [cx, cy, a, h, vcx, vcy, va, vh].
    embedding         : Current L2-normalised Re-ID embedding (EMA-updated).
    hits              : Frames this track has been matched. Gates 'tentative'
                        tracks — only hits >= min_hits IDs are drawn.
    time_since_update : Frames since last match. Reset to 0 on match.
                        Incremented every missed frame. Deleted at max_age.
    """

    def __init__(self, track_id: int, box: np.ndarray, embedding: np.ndarray):
        self.track_id = track_id
        self.kf = _make_kf(box)           # one KF per identity
        self.embedding = embedding.copy()
        self.hits: int = 1
        self.time_since_update: int = 0

    # ---- KF convenience wrappers -----------------------------------------
    def predict(self) -> None:
        """Advance KF one time-step (call once per frame, before matching)."""
        self.kf.predict()

    def kf_update(self, box: np.ndarray) -> None:
        """Correct KF with the matched detection bounding box."""
        self.kf.update(_box_to_z(box).reshape(4, 1))

    @property
    def predicted_box(self) -> np.ndarray:
        """Return the KF-predicted bounding box  [x1,y1,x2,y2]."""
        return _z_to_box(self.kf.x.flatten())


# ---------------------------------------------------------------------------
# Tracker  (DeepSORT-style: KF motion prediction + fused cost matrix)
# ---------------------------------------------------------------------------
class Tracker:
    """
    DeepSORT-style multi-object tracker.

    Each track owns a Kalman Filter for motion prediction so the tracker can
    bridge gaps during occlusions using physics rather than just appearance.

    Per-frame pipeline
    ------------------
    1. **KF predict** — all active tracks advance their KF one time-step,
       producing a predicted bounding box for this frame.
    2. **Age** — all tracks' `time_since_update` is incremented.
    3. **Fused cost matrix** — built from both appearance (cosine distance
       between ReID embeddings) and motion (1 - IoU between the KF-predicted
       bbox and each detected bbox).  Cost = α·appearance + (1-α)·motion.
    4. **Hungarian assignment** — globally optimal 1-to-1 matching.
    5. **Match gate** — reject pairs whose combined cost exceeds the
       appearance gate (cosine similarity < sim_threshold) OR whose IoU is
       zero (predicted bbox doesn't overlap detection at all).
    6. **KF update** — matched tracks correct their KF with the measurement.
    7. **EMA embedding update** — matched tracks blend their ReID embedding.
    8. **New tracks** — truly unmatched detections spawn tentative tracks.
    9. **Prune** — hard-delete tracks with time_since_update >= max_age.

    Lifecycle (unchanged from previous version)
    -------------------------------------------
    - Tentative tracks (hits < min_hits) → ID returned as -1, not drawn.
    - Dead tracks (time_since_update >= max_age) → hard-deleted from self.tracks.
    """

    def __init__(
        self,
        sim_threshold: float = 0.85,
        ema_alpha: float = 0.90,
        max_age: int = 30,
        min_hits: int = 3,
        appearance_weight: float = 0.5,
    ):
        """
        Parameters
        ----------
        sim_threshold     : Minimum cosine similarity to accept a match (0.85).
        ema_alpha         : EMA weight for Re-ID embedding updates (0.9).
        max_age           : Consecutive missed frames before track deletion (30).
        min_hits          : Matched frames before a track is drawn (3).
        appearance_weight : α in  cost = α·cosine_dist + (1-α)·(1-IoU).
                            0 → pure IoU/motion;  1 → pure appearance.
                            Default 0.5 = equal blend.
        """
        self.tracks: dict[int, Track] = {}
        self.next_id: int = 1
        self.sim_threshold = sim_threshold
        self.ema_alpha = ema_alpha
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight   # α

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        boxes: np.ndarray,
        embeddings: list[np.ndarray],
    ) -> list[int]:
        """
        Process one frame: predict, match, update, spawn, prune.

        Parameters
        ----------
        boxes      : (N, 4) detected bounding boxes in [x1,y1,x2,y2] format.
        embeddings : N Re-ID embeddings, one per detection.

        Returns
        -------
        list[int]  : One track ID per detection.
                     -1 means the track is still tentative (< min_hits).
        """
        n = len(embeddings)
        boxes = np.array(boxes, dtype=float) if n > 0 else np.empty((0, 4))

        # ── Step 1 & 2: KF predict + age ALL tracks ──────────────────────
        # predict() projects each track's state forward one time-step,
        # giving a kinematic estimate of where the person is *this* frame.
        for track in self.tracks.values():
            track.predict()                  # advance KF
            track.time_since_update += 1     # presumed lost until matched

        assigned = [-1] * n

        # matched_det_indices: detections associated to an existing track.
        # A tentative match also leaves assigned[i]==-1, so we cannot use
        # assigned[i]==-1 alone to decide whether to spawn a new track.
        matched_det_indices: set[int] = set()

        # ── Step 3: Build fused cost matrix ──────────────────────────────
        if self.tracks and n > 0:
            track_ids = list(self.tracks.keys())

            # --- Appearance cost (cosine distance, N_det × N_track) ---
            track_embs = np.stack([self.tracks[k].embedding for k in track_ids])
            det_embs   = np.stack(embeddings)
            app_cost   = cdist(det_embs, track_embs, metric="cosine")  # in [0,2]
            # Clamp to [0,1] — cosine dist can exceed 1 for anti-parallel vecs
            app_cost   = np.clip(app_cost, 0.0, 1.0)

            # --- Motion cost (1 - IoU between det_box and KF prediction) ---
            pred_boxes = np.stack(
                [self.tracks[k].predicted_box for k in track_ids]
            )  # (N_track, 4)
            iou_mat    = _iou_matrix(boxes, pred_boxes)  # (N_det, N_track)
            motion_cost = 1.0 - iou_mat                  # in [0,1]

            # --- Fused cost: α·appearance + (1-α)·motion ----------------
            α = self.appearance_weight
            cost = α * app_cost + (1.0 - α) * motion_cost

            # ── Step 4: Hungarian assignment ─────────────────────────────
            row_ind, col_ind = linear_sum_assignment(cost)

            # ── Step 5: Gate — reject poor matches ───────────────────────
            for r, c in zip(row_ind, col_ind):
                similarity   = 1.0 - app_cost[r, c]   # cosine similarity
                iou_val      = iou_mat[r, c]

                # Reject if appearance too dissimilar OR no spatial overlap.
                # (IoU gate prevents re-ID hallucinating across the frame.)
                if similarity < self.sim_threshold or iou_val <= 0.0:
                    continue

                tid   = track_ids[c]
                track = self.tracks[tid]

                # ── Step 6: KF measurement update ────────────────────────
                # Correct the KF with the detector's measurement this frame.
                track.kf_update(boxes[r])

                # ── Step 7: EMA embedding update ─────────────────────────
                track.embedding = (
                    self.ema_alpha * track.embedding
                    + (1.0 - self.ema_alpha) * embeddings[r]
                )
                track.hits += 1
                track.time_since_update = 0  # reset age — track is alive

                # Prevent Step 8 from spawning a duplicate track
                matched_det_indices.add(r)

                # Expose ID only once track is confirmed
                if track.hits >= self.min_hits:
                    assigned[r] = tid

        # ── Step 8: Spawn new tracks for truly unmatched detections ──────
        for i in range(n):
            if i not in matched_det_indices:
                new_track = Track(self.next_id, boxes[i], embeddings[i])
                self.tracks[self.next_id] = new_track
                if self.min_hits <= 1:
                    assigned[i] = self.next_id
                self.next_id += 1

        # ── Step 9: Hard-delete dead tracks ──────────────────────────────
        # Once removed from self.tracks, the KF and embedding are gone.
        # The Hungarian matrix shrinks — dead ghosts cannot steal matches.
        dead_ids = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update >= self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        return assigned

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------
    @property
    def gallery(self) -> dict:
        """Compat shim — {track_id: embedding} for active tracks."""
        return {tid: t.embedding for tid, t in self.tracks.items()}

    @property
    def confirmed_count(self) -> int:
        """Active tracks that have passed min_hits."""
        return sum(1 for t in self.tracks.values() if t.hits >= self.min_hits)

    @property
    def total_ids_issued(self) -> int:
        """Total unique IDs ever created."""
        return self.next_id - 1


# ---------------------------------------------------------------------------
# ReID transform
# ---------------------------------------------------------------------------
_reid_tf = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _id_to_bgr(track_id: int) -> tuple[int, int, int]:
    """Knuth multiplicative hash → visually distinct BGR colour per ID."""
    h = (track_id * 2654435761) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def _extract_embeddings(
    frame_rgb: np.ndarray,
    boxes: np.ndarray,
    embed_model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Crop detections, run through Siamese Network, L2-normalise."""
    H, W = frame_rgb.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        patch = frame_rgb[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((64, 32, 3), dtype=np.uint8)
        crops.append(_reid_tf(Image.fromarray(patch)))

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                embs = embed_model(batch)
        else:
            embs = embed_model(batch)
    return embs.cpu().numpy()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(
    device: torch.device,
    detector_weights: str = "fasterrcnn_mot16_finetuned.pth",
    reid_weights: str = "siamese_reid_mot16.pth",
) -> tuple[nn.Module, nn.Module]:
    """Load the fine-tuned Faster R-CNN and Siamese ReID model."""

    # --- Detector ---
    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    detector.load_state_dict(
        torch.load(detector_weights, map_location=device, weights_only=True)
    )
    detector.eval().to(device)

    # --- Re-ID ---
    embed_model = Siamese_Network(embed_dim=256)
    embed_model.load_state_dict(
        torch.load(reid_weights, map_location=device, weights_only=True)
    )
    embed_model.eval().to(device)

    return detector, embed_model


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def process_video(
    input_path: str,
    output_path: str,
    detector: nn.Module,
    embed_model: nn.Module,
    device: torch.device,
    det_thresh: float = 0.80,
    sim_thresh: float = 0.85,
    ema_alpha: float = 0.90,
    max_age: int = 30,
    min_hits: int = 3,
    output_fps: int = 25,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Run detection + tracking on *input_path* and write annotated video to
    *output_path*.

    Parameters
    ----------
    max_age  : Frames a track can be unmatched before being deleted (default 30).
    min_hits : Frames a track must be matched before its ID is drawn (default 3).
    progress_callback : callable(progress: float, message: str)
        Called once per frame with progress in [0, 1] and a status string.

    Returns
    -------
    dict  with keys: total_frames, unique_ids, active_ids, elapsed_seconds, avg_fps
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (W_out, H_out))

    tracker = Tracker(
        sim_threshold=sim_thresh,
        ema_alpha=ema_alpha,
        max_age=max_age,
        min_hits=min_hits,
    )

    t0 = time.time()

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Detection ---
        img_t = TF.to_tensor(frame_rgb).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    preds = detector([img_t])[0]
            else:
                preds = detector([img_t])[0]

        scores = preds["scores"].cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()
        boxes = boxes[scores >= det_thresh]

        # --- Tracking (KF predict happens inside tracker.update) ---
        if len(boxes) > 0:
            embs = _extract_embeddings(frame_rgb, boxes, embed_model, device)
            # Pass BOTH boxes and embeddings — KF needs the bbox measurement.
            tids = tracker.update(boxes, list(embs))
        else:
            # Even with no detections, we must still age + predict all tracks.
            tracker.update(np.empty((0, 4)), [])
            tids = []

        # --- Annotate confirmed tracks only (tid == -1 → tentative, skip) ---
        for box, tid in zip(boxes, tids):
            if tid == -1:
                # Track is tentative (< min_hits matches) — don't draw yet
                continue
            x1, y1, x2, y2 = map(int, box)
            color = _id_to_bgr(tid)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA,
            )

        writer.write(frame_bgr)
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(
                frame_idx / total_frames,
                f"Processing frame {frame_idx}/{total_frames} "
                f"· {len(boxes)} detections "
                f"· {tracker.confirmed_count} active / "
                f"{tracker.total_ids_issued} total IDs",
            )

    cap.release()
    writer.release()

    elapsed = time.time() - t0

    return {
        "total_frames": frame_idx,
        # unique_ids = all IDs ever issued (including pruned/dead tracks)
        "unique_ids": tracker.total_ids_issued,
        # active_ids = tracks still alive at the end of the video
        "active_ids": tracker.confirmed_count,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 1) if elapsed > 0 else 0,
    }

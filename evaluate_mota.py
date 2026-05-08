"""
evaluate_mota.py — Compute MOT metrics (MOTA, MOTP, ID switches, etc.)
                    for the TrackVision pipeline on MOT16 training sequences.

Usage:
    python evaluate_mota.py                           # all training sequences
    python evaluate_mota.py --sequences MOT16-09      # single sequence (fast)
    python evaluate_mota.py --sequences MOT16-02 MOT16-09  # specific sequences

Requirements:
    pip install motmetrics

Notes:
    - Only MOT16 TRAINING sequences have ground truth (test sequences do not).
    - Runs on CPU by default — no GPU required.
    - On CPU, expect ~1-3 FPS. Short sequences (MOT16-09: 525 frames) take
      a few minutes; long ones (MOT16-04: 1050 frames) take longer.
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from filterpy.kalman import KalmanFilter

try:
    import motmetrics as mm
except ImportError:
    print("ERROR: motmetrics not installed. Run:  pip install motmetrics")
    sys.exit(1)


# ── Ground-truth parser (same as notebook) ─────────────────────────────────
def parse_gt_file(file_path):
    """Parse MOT16 ground-truth file, returning {frame: [{id, bbox}, ...]}."""
    annotations = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0])
            obj_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf = int(parts[6])
            cls = int(parts[7])
            if conf != 1 or cls != 1:
                continue
            annotations[frame].append({
                "id": obj_id,
                "bbox": [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height],
            })
    return annotations


# ── Siamese Network (identical to training definition) ─────────────────────
class Siamese_Network(nn.Module):
    """Custom 3-layer CNN → 256-dim L2-normalised embedding."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 8, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


# ── Kalman Filter helpers (DeepSORT constant-velocity model) ─────────────────
default_val = None

def _box_to_z(box):
    """[x1,y1,x2,y2] → measurement [cx, cy, a, h]."""
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    return np.array([x1 + w/2, y1 + h/2, w/h if h > 0 else 1.0, h], dtype=float)

def _z_to_box(z):
    """KF state [cx, cy, a, h, ...] → [x1,y1,x2,y2]."""
    cx, cy, a, h = z[:4]
    w = a * h
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=float)

def _make_kf(box):
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8, dtype=float)
    for i in range(4): kf.F[i, i+4] = 1.0
    kf.H = np.zeros((4, 8), dtype=float); kf.H[:4, :4] = np.eye(4)
    kf.R *= 10.0; kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0; kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01; kf.Q[4:, 4:] *= 0.01
    kf.x[:4] = _box_to_z(box).reshape(4, 1)
    return kf

def _iou_matrix(boxes_a, boxes_b):
    M, N = len(boxes_a), len(boxes_b)
    iou = np.zeros((M, N), dtype=float)
    for i, a in enumerate(boxes_a):
        xx1 = np.maximum(a[0], boxes_b[:, 0]); yy1 = np.maximum(a[1], boxes_b[:, 1])
        xx2 = np.minimum(a[2], boxes_b[:, 2]); yy2 = np.minimum(a[3], boxes_b[:, 3])
        inter = np.maximum(0., xx2-xx1) * np.maximum(0., yy2-yy1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (boxes_b[:,2]-boxes_b[:,0])*(boxes_b[:,3]-boxes_b[:,1])
        union = area_a + area_b - inter
        iou[i] = np.where(union > 0, inter/union, 0.0)
    return iou


# ── Track — per-identity state object (KF-aware) ──────────────────────────
class Track:
    def __init__(self, track_id, box, embedding):
        self.track_id = track_id
        self.kf = _make_kf(box)
        self.embedding = embedding.copy()
        # last_box holds the most-recent raw detection; used as the position
        # estimate when KF is disabled (--no-kf ablation mode).
        self.last_box = np.array(box, dtype=float)
        self.hits = 1
        self.time_since_update = 0

    def predict(self):         self.kf.predict()
    def kf_update(self, box):  self.kf.update(_box_to_z(box).reshape(4, 1))

    @property
    def predicted_box(self):   return _z_to_box(self.kf.x.flatten())


# ── Tracker (DeepSORT-style: KF + fused cost matrix + lifecycle) ──────────
class Tracker:
    def __init__(self, sim_threshold=0.85, ema_alpha=0.9, max_age=30,
                 min_hits=3, appearance_weight=0.5, use_kalman=True):
        self.tracks = {}
        self.next_id = 1
        self.sim_threshold = sim_threshold
        self.ema_alpha = ema_alpha
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.use_kalman = use_kalman

    def update(self, boxes, embeddings):
        """
        boxes      : (N,4) [x1,y1,x2,y2]
        embeddings : N Re-ID embeddings
        Returns    : list of N track IDs (-1 = tentative)
        """
        n = len(embeddings)
        boxes = np.array(boxes, dtype=float) if n > 0 else np.empty((0, 4))

        # Step 1+2: KF predict + age all tracks.
        # When KF is disabled we still increment time_since_update — track
        # deletion logic is independent of the motion model.
        for t in self.tracks.values():
            if self.use_kalman:
                t.predict()
            t.time_since_update += 1

        assigned = [-1] * n
        matched_det_indices = set()

        # Step 3: Build fused cost matrix
        if self.tracks and n > 0:
            track_ids  = list(self.tracks.keys())
            track_embs = np.stack([self.tracks[k].embedding for k in track_ids])
            det_embs   = np.stack(embeddings)
            app_cost   = np.clip(cdist(det_embs, track_embs, metric="cosine"), 0., 1.)
            # KF mode: use KF-predicted box (motion-extrapolated position).
            # No-KF mode: use last observed box — no temporal extrapolation,
            # so IoU drops faster when a track is missed for several frames.
            if self.use_kalman:
                pred_boxes = np.stack([self.tracks[k].predicted_box for k in track_ids])
            else:
                pred_boxes = np.stack([self.tracks[k].last_box for k in track_ids])
            iou_mat    = _iou_matrix(boxes, pred_boxes)
            motion_cost = 1.0 - iou_mat
            α = self.appearance_weight
            cost = α * app_cost + (1.0 - α) * motion_cost

            # Step 4+5: Hungarian + gate
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if (1.0 - app_cost[r, c]) < self.sim_threshold: continue
                if iou_mat[r, c] <= 0.0: continue

                tid = track_ids[c]
                t   = self.tracks[tid]
                # Step 6: update position.
                # KF mode: correct the filter with the new measurement.
                # No-KF mode: just record the raw box for next frame's IoU cost.
                if self.use_kalman:
                    t.kf_update(boxes[r])
                t.last_box = boxes[r].copy()
                t.embedding = self.ema_alpha * t.embedding + (1-self.ema_alpha) * embeddings[r]
                t.hits += 1
                t.time_since_update = 0
                matched_det_indices.add(r)
                if t.hits >= self.min_hits:
                    assigned[r] = tid

        # Step 8: new tracks for unmatched detections
        for i in range(n):
            if i not in matched_det_indices:
                self.tracks[self.next_id] = Track(self.next_id, boxes[i], embeddings[i])
                if self.min_hits <= 1:
                    assigned[i] = self.next_id
                self.next_id += 1

        # Step 9: hard-delete dead tracks
        for tid in [k for k, t in self.tracks.items() if t.time_since_update >= self.max_age]:
            del self.tracks[tid]

        return assigned

    @property
    def gallery(self):
        return {tid: t.embedding for tid, t in self.tracks.items()}


# ── ReID transform ────────────────────────────────────────────────────────
_reid_tf = transforms.Compose([
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def extract_embeddings(frame_rgb, boxes, embed_model, device):
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
    return torch.nn.functional.normalize(embs, dim=1).cpu().numpy()


# ── IoU computation ───────────────────────────────────────────────────────
def compute_iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix between ground truth and predicted boxes."""
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    iou_matrix = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            x1 = max(gt[0], pred[0])
            y1 = max(gt[1], pred[1])
            x2 = min(gt[2], pred[2])
            y2 = min(gt[3], pred[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
            area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
            union = area_gt + area_pred - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0
    return iou_matrix


# ── Load models ───────────────────────────────────────────────────────────
def load_models(device, detector_path, reid_path):
    """Load fine-tuned Faster R-CNN and Siamese ReID model."""
    # Detector
    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.eval().to(device)

    # ReID
    embed_model = Siamese_Network(embed_dim=256)
    embed_model.load_state_dict(torch.load(reid_path, map_location=device))
    embed_model.eval().to(device)

    return detector, embed_model


# ── Evaluate one sequence ─────────────────────────────────────────────────
def evaluate_sequence(seq_name, mot16_root, detector, embed_model, device,
                      det_thresh, sim_thresh, ema_alpha, iou_thresh=0.5,
                      use_kalman=True):
    """Run tracker on a sequence and compute MOT metrics."""
    seq_path = os.path.join(mot16_root, "train", seq_name)
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    img_dir = os.path.join(seq_path, "img1")

    if not os.path.exists(gt_path):
        print(f"  ⚠ Skipping {seq_name} — no ground truth found")
        return None

    # Parse ground truth
    gt_annotations = parse_gt_file(gt_path)

    # Get frame paths
    frame_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    if len(frame_paths) == 0:
        print(f"  ⚠ Skipping {seq_name} — no frames found")
        return None

    # Initialize tracker
    tracker = Tracker(sim_threshold=sim_thresh, ema_alpha=ema_alpha,
                      use_kalman=use_kalman)

    # motmetrics accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    total_frames = len(frame_paths)
    t0 = time.time()

    for idx, frame_path in enumerate(frame_paths):
        frame_num = idx + 1

        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Detect ──
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

        # ── Track ──
        if len(boxes) > 0:
            embs = extract_embeddings(frame_rgb, boxes, embed_model, device)
            # Pass boxes so KF can init and update with real measurements
            tids = tracker.update(boxes, list(embs))
        else:
            # No detections: still must age + predict all active tracks
            tracker.update(np.empty((0, 4)), [])
            tids = []

        # ── Build GT / Hypothesis for this frame ──
        gt_frame = gt_annotations.get(frame_num, [])
        gt_ids = [g["id"] for g in gt_frame]
        gt_boxes_frame = [g["bbox"] for g in gt_frame]

        # Filter out tentative tracks (tid == -1) before feeding motmetrics.
        # Tentative detections have fewer than min_hits matches and must not
        # contribute to FP or TP counts — they are not committed hypotheses.
        confirmed = [
            (tid, list(b))
            for tid, b in zip(tids, boxes)
            if tid != -1
        ]
        pred_ids = [p[0] for p in confirmed]
        pred_boxes_frame = [p[1] for p in confirmed]

        # ── Compute distance matrix (1 - IoU) ──
        if len(gt_boxes_frame) > 0 and len(pred_boxes_frame) > 0:
            iou_mat = compute_iou_matrix(
                np.array(gt_boxes_frame), np.array(pred_boxes_frame)
            )
            dist_mat = 1.0 - iou_mat
            # Mask entries below IoU threshold
            dist_mat[iou_mat < iou_thresh] = np.nan
        else:
            dist_mat = np.empty((len(gt_ids), len(pred_ids)))
            dist_mat[:] = np.nan

        acc.update(gt_ids, pred_ids, dist_mat)

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == total_frames:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  {seq_name}: frame {idx + 1}/{total_frames}  "
                  f"({fps:.1f} FPS)  {len(boxes)} dets  "
                  f"{len(tracker.gallery)} IDs", end="\r")

    elapsed = time.time() - t0
    print(f"  {seq_name}: {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames / elapsed:.1f} FPS)  "
          f"{len(tracker.gallery)} unique IDs        ")

    return acc


# ── Tee: write to stdout and a file simultaneously ────────────────────────
class _Tee:
    """Replaces sys.stdout when --output is given, mirroring all print() calls
    to both the terminal and the output file. Carriage-return-only lines (the
    per-frame progress updates that overwrite each other in the terminal) are
    suppressed in the file so it stays readable."""
    def __init__(self, file, stdout):
        self._file = file
        self._stdout = stdout
    def write(self, data):
        self._stdout.write(data)
        # \r-only lines overwrite in terminal but pile up verbatim in files
        if not (data.endswith('\r') and '\n' not in data):
            self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MOT metrics on MOT16 training sequences."
    )
    parser.add_argument(
        "--mot16-root", default="./MOT16",
        help="Path to MOT16 dataset root (default: ./MOT16)"
    )
    parser.add_argument(
        "--sequences", nargs="+",
        default=["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09",
                 "MOT16-10", "MOT16-11", "MOT16-13"],
        help="Training sequences to evaluate (default: all 7)"
    )
    parser.add_argument(
        "--detector-weights", default="fasterrcnn_mot16_finetuned.pth",
        help="Path to detector weights"
    )
    parser.add_argument(
        "--reid-weights", default="siamese_reid_mot16.pth",
        help="Path to ReID weights"
    )
    parser.add_argument("--det-thresh", type=float, default=0.80)
    parser.add_argument("--sim-thresh", type=float, default=0.85)
    parser.add_argument("--ema-alpha", type=float, default=0.90)
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for matching (default: 0.5)")
    parser.add_argument("--output", default=None,
                        help="Save results to this file in addition to stdout")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --output file if it already exists")
    parser.add_argument("--no-kf", action="store_true",
                        help="Disable Kalman filter (ablation: pure IoU + appearance matching, no motion prediction)")

    args = parser.parse_args()

    if args.output and os.path.exists(args.output) and not args.force:
        print(f"[!] Output file already exists: {args.output}")
        print("    Use --force to overwrite.")
        sys.exit(1)

    # ── Reconfigure stdout encoding ──
    sys.stdout.reconfigure(encoding="utf-8")

    _outfile = None
    try:
        if args.output:
            _outfile = open(args.output, "w", encoding="utf-8")
            sys.stdout = _Tee(_outfile, sys.stdout)

        # ── Device ──
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("Device: CPU (no GPU — this will be slow but works fine)")

        # ── Load models ──
        print("Loading models...")
        detector, embed_model = load_models(
            device, args.detector_weights, args.reid_weights
        )
        use_kalman = not args.no_kf
        mode_str = "WITH Kalman filter" if use_kalman else "WITHOUT Kalman filter (--no-kf ablation)"
        print(f"Models loaded ✓\n")
        print(f"Tracking mode: {mode_str}\n")

        # ── Evaluate each sequence ──
        accumulators = []
        seq_names = []

        for seq in args.sequences:
            print(f"Evaluating {seq}...")
            acc = evaluate_sequence(
                seq, args.mot16_root, detector, embed_model, device,
                args.det_thresh, args.sim_thresh, args.ema_alpha, args.iou_thresh,
                use_kalman=use_kalman,
            )
            if acc is not None:
                accumulators.append(acc)
                seq_names.append(seq)

        if not accumulators:
            print("No sequences evaluated.")
            return

        # ── Compute metrics ──
        print("\n" + "=" * 70)
        print("MOT EVALUATION RESULTS")
        print("=" * 70)

        mh = mm.metrics.create()

        # Per-sequence metrics
        summary = mh.compute_many(
            accumulators, names=seq_names,
            metrics=["num_frames", "mota", "motp", "num_switches",
                     "num_false_positives", "num_misses",
                     "precision", "recall", "idf1"],
            generate_overall=True,
        )

        # Rename columns for readability
        summary.columns = ["Frames", "MOTA", "MOTP", "ID Sw.",
                            "FP", "FN", "Prec.", "Recall", "IDF1"]

        print(mm.io.render_summary(summary, namemap={}, formatters=mh.formatters))

        # ── Extract headline numbers ──
        overall = summary.loc["OVERALL"]
        mota_pct = overall["MOTA"] * 100
        id_switches = int(overall["ID Sw."])
        n_seqs = len(seq_names)

        print("\n" + "-" * 70)
        print("SUMMARY:")
        print(f'  MOTA: {mota_pct:.1f}%  |  ID Switches: {id_switches}  |  Sequences: {n_seqs}')
        print("-" * 70)

    finally:
        if _outfile:
            # restore sys.stdout first, otherwise Python's shutdown tries to flush the _Tee
            sys.stdout = sys.stdout._stdout
            _outfile.close()


if __name__ == "__main__":
    main()

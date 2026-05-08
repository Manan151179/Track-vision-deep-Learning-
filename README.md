# TrackVision — Multi-Object Pedestrian Tracking on MOT16

End-to-end pedestrian tracking system benchmarked on the MOT16 dataset.
Deployed as a Streamlit web app and a real-time Colab webcam demo.

**Overall MOTA 0.661 · IDF1 0.624 · 1,854 ID switches across 7 sequences**

---

## Overview

TrackVision detects and tracks every pedestrian in a video, assigning each
person a persistent ID that survives occlusions and crowded scenes. It is
built on three components trained end-to-end on the MOT16 benchmark:

1. **Detector** — Faster R-CNN (ResNet-50 FPN) fine-tuned on MOT16 for
   2-class detection (background / pedestrian). 41.3 M parameters.
2. **Re-ID model** — Lightweight 3-layer CNN that maps 128×64 person crops
   to 256-D L2-normalised embeddings. 8.6 M parameters, trained with
   triplet margin loss.
3. **Tracker** — DeepSORT-style tracker: per-track Kalman filter (8-D
   constant-velocity model) for motion prediction, Hungarian assignment on a
   fused appearance (cosine distance) + motion (1 − IoU) cost matrix, and
   EMA-blended embedding gallery for long-term identity memory.

---

## Pipeline

For each video frame the system runs: Faster R-CNN produces bounding boxes
with confidence scores, which are thresholded at `DET_THRESH` (default 0.80).
Each surviving crop is embedded by the Siamese network into a 256-D
L2-normalised vector. The tracker predicts every active track's location
forward one time-step via its Kalman filter, builds a fused cost matrix
(α · cosine distance + (1−α) · (1−IoU)), solves the globally-optimal
assignment with the Hungarian algorithm, gates matches on both cosine
similarity ≥ `SIM_THRESH` and IoU > 0, updates matched tracks (KF
correction + EMA embedding blend), ages unmatched tracks, and hard-deletes
any track that has been unmatched for `max_age` consecutive frames.
Tracks confirmed after `min_hits` matches are drawn on the output frame.

---

## Quick Start

### 1. Clone and pull weights

```bash
git clone https://github.com/Manan151179/Track-vision-deep-Learning-.git
cd Track-vision-deep-Learning-

# Model weights are stored in Git LFS (~200 MB total)
git lfs pull
```

> If `git lfs` is not installed: `brew install git-lfs` (macOS) or
> `sudo apt install git-lfs` (Ubuntu), then `git lfs install`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install "pandas<2.0" "numpy<2.0"   # required locally — motmetrics 1.4.0 is incompatible with pandas 2+
```

This installs: `torch`, `torchvision`, `streamlit`, `opencv-python`,
`scipy`, `Pillow`, `numpy`, `filterpy`, `motmetrics`.

**Also install ffmpeg** (system binary — not a pip package):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

ffmpeg is required by the Streamlit app to re-encode output video for
browser playback.

### 3. Verify the install

```bash
python -c "import torch, torchvision, streamlit, cv2, scipy, PIL, numpy, filterpy, motmetrics; print('OK')"
```

Expected output: `OK`

### 4. Get the MOT16 dataset

Download MOT16 from [motchallenge.net](https://motchallenge.net/data/MOT16/)
and place it at `./MOT16/` so the structure looks like:

```
MOT16/
├── train/
│   ├── MOT16-02/
│   │   ├── img1/   ← 000001.jpg … 000600.jpg (1-indexed)
│   │   └── gt/gt.txt
│   ├── MOT16-04/
│   └── … (MOT16-05, 09, 10, 11, 13)
└── test/
    ├── MOT16-01/
    └── … (MOT16-03, 06, 07, 08, 12, 14)
```

The `MOT16/` directory is gitignored — you must download it separately.
The dataset is only required for evaluation. The web app works on any
uploaded video without the dataset.

---

## Running the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

Upload an MP4/AVI/MOV/MKV video, adjust the tracking parameters in the
sidebar if desired, and click **Run Tracking**. The app processes the video
frame-by-frame (with a live progress bar), displays result metrics, plays
the annotated output inline, and offers a one-click download.

**Sidebar parameters and their defaults:**

| Parameter | Default | Effect |
|---|---|---|
| Detection threshold | 0.80 | Minimum detector confidence to keep a box |
| Similarity threshold | 0.85 | Minimum cosine similarity to match a detection to an existing track |
| EMA smoothing α | 0.90 | Weight of the existing embedding in the EMA gallery update |
| Max track age | 30 frames | Consecutive missed frames before a track is deleted |
| Min hits to confirm | 3 frames | Matched frames before a track ID is drawn |
| Output FPS | 25 | Frame rate of the exported video |

**Device selection** is automatic: CUDA → CPU (MPS is disabled — see Known
Limitations).

---

## Running Evaluation

Evaluation requires the MOT16 training sequences and `motmetrics`.
It re-runs the full tracker on each sequence and compares predictions against
ground truth.

```bash
# All 7 training sequences (~10 min GPU / ~30–60 min CPU)
python evaluate_mota.py

# Single fast sequence (~1 min GPU / ~3 min CPU)
python evaluate_mota.py --sequences MOT16-09

# Two specific sequences
python evaluate_mota.py --sequences MOT16-02 MOT16-11

# Override inference thresholds
python evaluate_mota.py --det-thresh 0.75 --sim-thresh 0.80 --iou-thresh 0.5

# Save output to a file (stdout is still printed; --force to overwrite existing)
python evaluate_mota.py --output eval_results.txt
```

All `argparse` defaults match the web app defaults. The script prints a
formatted metrics table (MOTA, MOTP, ID switches, FP, FN, Precision,
Recall, IDF1) and a headline summary line.

Pre-computed results are in `eval_results-full.txt`.

---

## Training (Google Colab)

Training runs on Google Colab Pro with a GPU runtime (A100 recommended).
**The notebook is hardcoded for Colab Google Drive paths and will not run
locally without edits** — see Known Limitations.

1. Upload the MOT16 dataset to `MyDrive/MOT16/` on your Google Drive.
2. Open `deep_learning.ipynb` on Colab and select a GPU runtime
   (Runtime → Change runtime type → A100/V100).
3. Run cells in order:

| Cells | Stage | Duration (A100) |
|---|---|---|
| 1–22 | Data preparation and sanity check | ~2 min |
| 23–27 | Faster R-CNN fine-tuning (5 epochs, SGD lr=0.005, AMP) | ~20 min |
| 28–36 | Siamese ReID training (10 epochs, Adam lr=1e-3, triplet margin=1.0, AMP) | ~15 min |
| 37–45 | Inference on a test sequence, saves `mot16_tracked.mp4` | ~5 min |

Saved weights:
- `fasterrcnn_mot16_finetuned.pth` (~165 MB, 41.3 M params)
- `siamese_reid_mot16.pth` (~34 MB, 8.6 M params)

Both are already included in the repo via Git LFS — training only needs to
be re-run to reproduce or change the weights.

---

## Project Layout

```
Deep_Learning/
├── tracker_engine.py          # Core inference: Siamese, Track, Tracker, Kalman helpers
├── app.py                     # Streamlit web UI; imports tracker_engine
├── evaluate_mota.py           # MOT benchmark evaluation (MOTA, MOTP, IDF1, …)
│
├── deep_learning.ipynb        # Full 4-stage training pipeline (Colab only)
├── visualizations.ipynb       # Dataset analysis and figure generation
├── realtime_colab.ipynb       # Live webcam tracking demo (Colab + browser JS)
├── deploy_colab.ipynb         # Runs Streamlit on Colab via ngrok/cloudflared tunnel
│
├── fasterrcnn_mot16_finetuned.pth   # Detector weights — Git LFS (~165 MB)
├── siamese_reid_mot16.pth           # ReID model weights — Git LFS (~34 MB)
│
├── eval_results-full.txt      # Saved MOT16 evaluation results
├── project_notes.txt          # Narrative design documentation
├── requirements.txt           # pip dependencies
│
└── MOT16/                     # Dataset — gitignored, download separately
```

---

## Known Limitations

- **Real-time performance is ~2–5 FPS** when running the webcam demo on a
  Colab A100 GPU. The bottleneck is the JavaScript-to-Python frame transfer
  roundtrip, not the GPU. The Streamlit app processes pre-recorded videos
  faster (~15–30 FPS on A100).

- **Training notebook is Colab-only.** `deep_learning.ipynb` cells 2 and 4
  are hardcoded for `/content/drive/MyDrive/MOT16` and call
  `google.colab.drive.mount()`. Running locally requires manually editing
  `CONFIG["MOT16_ROOT"]` and removing the Drive mount cell.

- **Tracker code is duplicated.** `evaluate_mota.py` contains its own full
  copies of `Siamese_Network`, `Track`, `Tracker`, all Kalman filter helpers,
  and `_reid_tf`. These are kept in sync manually with `tracker_engine.py`.
  A future refactor would have `evaluate_mota.py` import directly from
  `tracker_engine`.

- **Apple MPS is disabled.** PyTorch's MPS backend has stability issues with
  Faster R-CNN. The app and eval script use CUDA → CPU fallback only.

- **No train/validation split.** Both models are trained on all 7 MOT16
  training sequences. Overfitting cannot be detected from training logs alone.
  This is standard MOT practice (test sequences have no ground truth), but
  worth noting.

# Project Status

_Last updated: 2026-05-07_

## ✅ Completed features

- Stage 1: Data preparation — MOT16 GT parsing, MOTDataset class, ConcatDataset across 7 train sequences (5,316 frames). See `deep_learning.ipynb` Cells 1–22.
- Stage 2: Faster R-CNN fine-tuning — torchvision FasterRCNN R50-FPN, 2-class head (bg + pedestrian), backbone frozen, SGD lr=0.005, 5 epochs, AMP. Saved: `fasterrcnn_mot16_finetuned.pth`. Loss 0.62 → 0.48.
- Stage 3: Siamese ReID training — 3 conv blocks → FC → 256-D L2-normalised embedding, TripletMarginLoss margin=1.0, Adam lr=1e-3, 10 epochs, AMP. Saved: `siamese_reid_mot16.pth`.
- Stage 4: DeepSORT-style tracker — per-track Kalman filter (8-D constant velocity), Hungarian matching on weighted IoU + cosine appearance cost, EMA embedding smoothing, max_age + min_hits track lifecycle. See `tracker_engine.py`.
- Streamlit web UI — full inference with hyperparameter sliders, GPU/CPU/MPS auto-detection, ffmpeg H.264 re-encode, download. See `app.py`.
- MOT benchmark evaluation — motmetrics-based MOTA/MOTP/IDF1 on all 7 train sequences. Overall MOTA 0.661, IDF1 0.623. See `evaluate_mota.py` and `baseline_metrics.txt`.
- Sample tracked output video — `mot16_tracked.mp4`.
- Midterm presentation — delivered. Methodology now locked in for final.
- Project memory bootstrap — `CLAUDE.md`, `STATUS.md`, `CHANGELOG.md`.
- Fix `requirements.txt` + write `README.md` — added `filterpy`, uncommented `motmetrics`; full README with Quick Start, web app, eval, training, and known limitations sections.
- `generate_report_assets.py` — single script to produce `tracked.mp4`, `frame_grid.png`, `speed_stats.json` from any MOT16 sequence; imports from `tracker_engine.py` (no duplication); validated on MOT16-09 5-frame smoke test.

## 🔨 In progress

- Nothing currently in progress.

## 📋 Next up (planned but not started)

Final submission deliverables — deadline Sunday night:

1. ~~Fix `requirements.txt` (add `filterpy`, uncomment `motmetrics`).~~ ✅ Done.
2. ~~Write `README.md` with run instructions for both local and Colab.~~ ✅ Done.
3. ~~Script to generate report assets (tracked video + frame grid + speed stats).~~ ✅ Done (`generate_report_assets.py`).
4. Pick a target test pedestrian on a chosen MOT16 test sequence; tune inference thresholds (det / sim / EMA / max_age / min_hits); regenerate the submission demo video using `generate_report_assets.py`.
4. Generate report figures: training loss curves, qualitative tracked frames, inference threshold sensitivity table (counts as a safe ablation — no retraining needed).
5. Write 4+ page LaTeX report (Abstract + Intro, Methodology, Challenges, Self-evaluation + Citations).
6. Strip model weights from final submission package per rubric.

Optional / time-permitting:

- Deduplicate tracker code between `tracker_engine.py` and `evaluate_mota.py` (refactor `evaluate_mota.py` to import from `tracker_engine.py`).
- Reconcile or delete `project_notes.txt` (stale facts contradict code).
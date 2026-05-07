# Changelog

All notable changes to TrackVision, newest first.

## [2026-05-07] Add generate_report_assets.py

**What changed:**
- New script `generate_report_assets.py` at project root.
- Runs the tracker on any MOT16 sequence (from a frame directory) and produces
  three report artifacts in a configurable output directory:
  - `tracked.mp4` — full annotated video, H.264 re-encoded via ffmpeg for browser playback.
  - `frame_grid.png` — 2×3 grid of evenly-sampled annotated frames with "Frame N" labels (matplotlib when available; PIL fallback otherwise).
  - `speed_stats.json` — `{"sequence", "device", "total_frames", "total_seconds", "fps", "det_thresh", "sim_thresh"}`.
- Imports `load_models` and `process_video` from `tracker_engine.py` — no tracker code duplicated.
- Supports `--sequence`, `--output-dir`, `--num-frames`, `--det-thresh`, `--sim-thresh`, `--ema-alpha`, `--fps`.
- Clear `sys.exit()` messages for: missing sequence path, missing img1/, missing model weights, ffmpeg not on PATH.

**Why:**
Single command to produce all three report artifacts for local CPU validation and
full Colab GPU submission without changing any code between runs.

**Files touched:**
- `generate_report_assets.py` (new)

**Tests added:**
- Smoke test passed: 5 real frames from MOT16-09, CPU + MPS. All three output files
  produced and speed_stats.json is valid JSON. All three error-exit paths exercised.

**Out-of-scope observations:**
- `tracker_engine.py:559` has a bug: `tids` is not assigned when `len(boxes) == 0`,
  causing `UnboundLocalError` if ALL frames in a run have zero detections. This never
  fires on real MOT16 sequences (detector always finds pedestrians), but it was
  exposed by the all-black dummy frames smoke test. Fix: add `tids = []` to the
  `else` branch of the `if len(boxes) > 0` block. Deferred per standing rule 2.
- matplotlib is not installed in `trackvision_env/` (not in requirements.txt).
  The PIL fallback path in `generate_report_assets.py` handles this correctly.
  If a nicer grid is desired, `pip install matplotlib` into the venv.

---

## [2026-05-07] Fix requirements.txt and write README.md

**What changed:**
- `requirements.txt`: added `filterpy` (missing entirely); uncommented `motmetrics` (was commented out). Both are required at runtime by `tracker_engine.py` and/or `evaluate_mota.py`.
- `README.md`: new file at project root with sections: Overview, Pipeline, Quick Start (clone + lfs pull + pip install + ffmpeg + dataset), Running the Web App, Running Evaluation, Training (Colab), Project Layout, Known Limitations.

**Why:**
Fresh `pip install -r requirements.txt` crashed immediately on `from filterpy.kalman import KalmanFilter`. Missing README forced reverse-engineering to run anything. Both fix reproducibility for graders and collaborators.

**Files touched:**
- `requirements.txt` (modified — 2 lines changed)
- `README.md` (new)

**Tests added:**
- Import verification: `python -c "import torch, torchvision, streamlit, cv2, scipy, PIL, numpy, filterpy, motmetrics; print('OK')"` — confirms all runtime dependencies resolve after install.

**Out-of-scope observations:**
- `evaluate_mota.py` duplicates Siamese, Track, Tracker, and Kalman helpers from `tracker_engine.py` — deferred per CLAUDE.md standing rule 2.
- `project_notes.txt` contains stale facts (optimizer, margin) contradicting the notebook — deferred.

---

## [2026-05-07] Project memory bootstrap

**What changed:**
- Created `CLAUDE.md`, `STATUS.md`, and `CHANGELOG.md` at project root.
- Documented current pipeline, tech stack, commands, conventions, and known gotchas.

**Why:**
Project memory must exist before Claude Code can work in scope across sessions.

**Files touched:**
- `CLAUDE.md` (new)
- `STATUS.md` (new)
- `CHANGELOG.md` (new)

**Tests added:**
- N/A (documentation only).

**Out-of-scope observations:**
- `requirements.txt` is missing `filterpy` and has `motmetrics` commented out — both required at runtime. Will be fixed in next task.
- `evaluate_mota.py` duplicates Siamese, Track, Tracker, Kalman helpers from `tracker_engine.py` — deferred.
- `project_notes.txt` lists optimizer=AdamW and margin=0.5 but actual code uses SGD and margin=1.0 — deferred (likely safe to delete the file).
- `deep_learning.ipynb` cells 2 and 4 are hardcoded for Colab `/content/drive/MyDrive/MOT16` — out of scope for this session.
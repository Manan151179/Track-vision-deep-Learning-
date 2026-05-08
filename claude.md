# Project: TrackVision — Multi-Object Tracking on MOT16

## What this project is

TrackVision is a deep-learning multi-object tracking system that detects pedestrians in video and assigns each one a persistent identity across frames. Built for COMP-SCI 5567 Deep Learning final project. Pipeline: fine-tuned Faster R-CNN detects pedestrians → Siamese network produces 256-D appearance embeddings → DeepSORT-style tracker (Kalman filter + Hungarian matching on combined IoU motion + cosine appearance cost) maintains IDs across frames. Output: annotated MP4 with persistent IDs.

## Tech stack

- Language: Python 3.10 (effectively required — pandas 1.5.x has no wheels for 3.12+)
- Framework: PyTorch 2.x (uses torch.amp for mixed precision), torchvision (Faster R-CNN R50-FPN)
- Tracking: filterpy (Kalman filter), scipy.optimize.linear_sum_assignment (Hungarian matching)
- Evaluation: motmetrics
- UI: Streamlit
- Video I/O: opencv-python, ffmpeg (system binary, used for H.264 re-encode)
- Other: numpy<2.0, scipy, Pillow
- Compute: Google Colab Pro (GPU) for training; local machine for inference, eval, and Streamlit UI

Colab runs unpinned. Local requires `pandas<2.0` and `numpy<2.0` — motmetrics 1.4.0 is incompatible with pandas 2+.

## Commands

Local development
streamlit run app.py                                           # Launch web UI on http://localhost:8501
python evaluate_mota.py                                        # Eval on all 7 MOT16 train sequences
python evaluate_mota.py --sequences MOT16-09                   # Single sequence (~3 min GPU)
python evaluate_mota.py --det-thresh 0.75                      # Override defaults
python evaluate_mota.py --output results.txt                   # Save output to file (--force to overwrite)
Install
pip install -r requirements.txt                 # includes filterpy and motmetrics
pip install "pandas<2.0" "numpy<2.0"            # required locally for motmetrics compatibility
Colab training (open deep_learning.ipynb on Colab Pro GPU)
Cells 1-22:  data prep + sanity check
Cells 23-27: Faster R-CNN fine-tuning (~20 min on A100)
Cells 28-36: Siamese ReID training (~15 min on A100)
Cells 37-45: inference, saves mot16_tracked.mp4

## Architecture

Deep_Learning/
├── tracker_engine.py        # Core inference: Siamese class, Track, Tracker, Kalman helpers
├── app.py                   # Streamlit UI; imports tracker_engine
├── evaluate_mota.py         # MOT benchmark eval (DUPLICATES tracker code from tracker_engine.py)
├── deep_learning.ipynb      # Training notebook — Colab only (hardcoded /content/drive paths)
├── visualizations.ipynb     # Report figures
├── realtime_colab.ipynb     # Webcam demo
├── deploy_colab.ipynb       # Streamlit-via-tunnel on Colab
├── fasterrcnn_mot16_finetuned.pth   # Detector weights (Git LFS, ~165 MB)
├── siamese_reid_mot16.pth           # ReID weights (Git LFS, ~34 MB)
├── MOT16/                   # Dataset (gitignored; download from motchallenge.net)
└── vis_output/              # Generated figures

Data flow per frame: image → Faster R-CNN (boxes with score ≥ DET_THRESH) → Siamese (256-D L2-normalised embedding per box) → Tracker (Kalman predict → IoU+appearance cost matrix → Hungarian assignment → update matched, age unmatched, create new) → annotated frame → MP4 via OpenCV → ffmpeg H.264 re-encode for browser playback.

## Conventions

- Inference hyperparameters live in three places — keep them in sync: `tracker_engine.py` defaults, `app.py` Streamlit slider defaults, `evaluate_mota.py` argparse defaults.
- The Siamese model definition is duplicated across `tracker_engine.py`, `evaluate_mota.py`, and `deep_learning.ipynb`. The `tracker_engine.py` version applies `F.normalize` inside `forward()`; the other two do not (they normalise post-hoc). Be aware when porting weights or refactoring.
- Output videos default to OpenCV `mp4v` codec; always run an ffmpeg re-encode pass before sharing for browser playback. `app.py` does this automatically; the notebook does not.
- MOT16 frames are 1-indexed (000001.jpg = frame 1) in both filenames and ground truth.
- Metrics from `evaluate_mota.py` are saved via `--output` (e.g. `eval_results-full.txt`). Don't overwrite a prior results file without committing it first.

## Standing rules for Claude Code (DO NOT REMOVE)

1. STAY IN SCOPE. Do everything required to complete the assigned task across however many files necessary, but do NOT modify anything unrelated to the task.
   In scope examples:
   - Fixing a bug across all files the bug touches
   - Adding tests for the fix
   - Updating types affected by the fix
   Out of scope examples:
   - Refactoring nearby code that "could be cleaner"
   - Fixing unrelated typos or formatting
   - Upgrading dependencies because newer versions exist
   - "Improving" anything not explicitly asked for

2. REPORT, DON'T FIX. If you spot something genuinely worth fixing outside scope (real bug, security issue, broken pattern), report it in your final message. Do NOT silently change it.

3. PRODUCTION-LEVEL TESTS. Every feature must include tests that would survive in a production codebase: unit tests for logic, integration tests for flows, edge cases covered. Tests must pass before the task is complete.

4. UPDATE CHANGELOG.md. After completing each feature, append an entry using the format defined in CHANGELOG.md.

5. STOP WHEN DONE. Once the task is complete and tests pass, stop. Do not continue with adjacent improvements.

6. UPDATE STATUS.md. At the end of every session (whether the task completed fully or stopped partway), update STATUS.md to reflect: what's done, what's in progress (with exact file/line if mid-feature), and what's next.

## Watch out for

- `deep_learning.ipynb` Cells 2 and 4 are HARDCODED for Colab: `MOT16_ROOT = "/content/drive/MyDrive/MOT16"` and `from google.colab import drive; drive.mount(...)`. Notebook will not run locally without edits.
- `project_notes.txt` contains STALE facts that contradict the actual code. It says optimizer=AdamW lr=1e-4 (actual: SGD lr=0.005) and TripletMarginLoss margin=0.5 (actual: 1.0). Trust the notebook, never cite project_notes.txt numbers in the report.
- Tracker code is DUPLICATED between `tracker_engine.py` and `evaluate_mota.py` (Siamese class, Track, Tracker, Kalman helpers, _reid_tf transform). Editing one without the other will silently desync them.
- `mot16_tracked.mp4` from the notebook uses OpenCV `mp4v` codec — may not play in browsers. Streamlit app re-encodes with ffmpeg; notebook does not.
- No train/val split — both models train on all 7 training sequences. Cannot detect overfitting from train-time logs. Standard MOT setup (test has no GT), but worth knowing.
- Apple MPS device path exists in CONFIG but is unstable with Faster R-CNN per project history — don't assume MPS works on Mac.
- Final submission must NOT include model weights (`*.pth`) per rubric — they're in the repo via Git LFS but must be excluded for Canvas submission.
- Architecture is LOCKED IN as of midterm presentation. Do not swap detector, ReID model, or tracking algorithm — only tune hyperparameters and clean up.
- Final report deadline: Sunday night. Rubric requires ≥4 pages, font ≤11, sections: Abstract+Intro, Methodology, Challenges, Self-evaluation+Citations. LaTeX recommended (Overleaf).
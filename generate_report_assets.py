"""
generate_report_assets.py — Run the tracker on a MOT16 sequence and produce
three report artifacts:

  tracked.mp4       Full annotated tracking video (H.264 re-encoded via ffmpeg
                    for browser / report compatibility).
  frame_grid.png    2x3 grid of evenly-sampled annotated frames with Frame N
                    labels.
  speed_stats.json  {"sequence", "device", "total_frames", "total_seconds",
                    "fps", "det_thresh", "sim_thresh"}

Usage:
    python generate_report_assets.py \\
        --sequence MOT16/test/MOT16-03 \\
        --output-dir vis_output/MOT16-03 \\
        [--num-frames 30] \\
        [--det-thresh 0.80] \\
        [--sim-thresh 0.85] \\
        [--ema-alpha 0.90] \\
        [--fps 25]

When --num-frames is omitted the full sequence is processed.
All threshold defaults match INFER_CONFIG in deep_learning.ipynb.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import torch

from tracker_engine import load_models, process_video


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> None:
    """Exit with a helpful message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: ffmpeg not found on PATH.\n"
            "  macOS  :  brew install ffmpeg\n"
            "  Ubuntu :  sudo apt install ffmpeg"
        )


def _check_weights() -> tuple[str, str]:
    """Verify both model weight files exist. Returns (det_path, reid_path)."""
    root = Path(__file__).parent
    det  = root / "fasterrcnn_mot16_finetuned.pth"
    reid = root / "siamese_reid_mot16.pth"
    missing = [str(p) for p in (det, reid) if not p.exists()]
    if missing:
        sys.exit(
            "ERROR: model weight file(s) not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun:  git lfs pull"
        )
    return str(det), str(reid)


def _validate_sequence(seq_path: Path) -> list[Path]:
    """
    Confirm the sequence directory exists and contains img1/*.jpg frames.
    Returns a sorted list of frame Paths.
    """
    if not seq_path.exists():
        sys.exit(
            f"ERROR: sequence path not found: {seq_path}\n"
            "Ensure MOT16/ is present and --sequence points to a valid directory."
        )
    img_dir = seq_path / "img1"
    if not img_dir.is_dir():
        sys.exit(
            f"ERROR: expected img1/ subdirectory not found inside: {seq_path}\n"
            "A MOT16 sequence must contain img1/*.jpg frames."
        )
    frames = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        sys.exit(f"ERROR: no .jpg frames found in {img_dir}")
    return frames


# ---------------------------------------------------------------------------
# Temp video assembly
# ---------------------------------------------------------------------------

def _build_temp_video(frame_paths: list[Path], fps: int) -> str:
    """
    Write the selected JPEG frames into a temporary mp4v .mp4 file.
    process_video() expects a video path rather than a frame directory, so we
    assemble one here via OpenCV VideoWriter (same codec used by process_video).
    Returns the temp file path; caller is responsible for deletion.
    """
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        sys.exit(f"ERROR: cannot read frame {frame_paths[0]}")
    H, W = sample.shape[:2]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (W, H))
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return tmp.name


# ---------------------------------------------------------------------------
# H.264 re-encode (matches app.py pattern exactly)
# ---------------------------------------------------------------------------

def _ffmpeg_reencode(src: str, dst: str) -> None:
    """Re-encode src to H.264 dst for browser / report playback. Exits on error."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            dst,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: ffmpeg re-encode failed:\n{result.stderr[-600:]}")


# ---------------------------------------------------------------------------
# Frame grid
# ---------------------------------------------------------------------------

def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    """
    Return `count` indices evenly spaced in [0, total-1], always including both
    the first (0) and last (total-1) frame.
    """
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


def _extract_grid_frames(video_path: str, n: int = 6) -> list[tuple[int, object]]:
    """
    Seek to n evenly-spaced positions in video_path and read one frame each.
    Returns [(0-based frame index, bgr ndarray), ...].
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = _evenly_spaced_indices(total, n)
    result = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            result.append((idx, frame))
    cap.release()
    return result


def _save_frame_grid(frames: list[tuple[int, object]], out_path: str) -> None:
    """
    Render a 2x3 grid of annotated frames labelled 'Frame N'.
    Tries matplotlib first; falls back to PIL if matplotlib is not installed.
    """
    rows, cols = 2, 3

    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive; must be set before pyplot import
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
        axes = axes.flatten()

        for ax, (idx, bgr) in zip(axes, frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"Frame {idx + 1}", fontsize=9, pad=4)
            ax.axis("off")

        # Hide any unfilled cells (e.g. when fewer than 6 frames are available)
        for ax in axes[len(frames):]:
            ax.axis("off")

        plt.tight_layout(pad=0.5)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        _save_frame_grid_pil(frames, out_path, rows, cols)


def _save_frame_grid_pil(
    frames: list[tuple[int, object]],
    out_path: str,
    rows: int,
    cols: int,
) -> None:
    """PIL implementation of the 2x3 frame grid (fallback when matplotlib absent)."""
    from PIL import Image, ImageDraw

    if not frames:
        return

    cell_w, cell_h = 480, 270
    label_h = 22
    pad = 6
    full_cell_h = cell_h + label_h + pad

    canvas = Image.new(
        "RGB",
        (cols * (cell_w + pad), rows * full_cell_h),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)

    for i, (idx, bgr) in enumerate(frames):
        row, col = divmod(i, cols)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cell_img = Image.fromarray(rgb).resize((cell_w, cell_h), Image.LANCZOS)
        x = col * (cell_w + pad)
        y = row * full_cell_h
        canvas.paste(cell_img, (x, y))
        draw.text(
            (x + 4, y + cell_h + 4),
            f"Frame {idx + 1}",
            fill=(210, 210, 210),
        )

    canvas.save(out_path)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate report assets from a MOT16 sequence: "
            "tracked.mp4, frame_grid.png, speed_stats.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence", required=True,
        help="Path to a MOT16 sequence directory (must contain img1/*.jpg).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output files (created if it does not exist).",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Process only the first N frames. Omit to run the full sequence.",
    )
    parser.add_argument(
        "--det-thresh", type=float, default=0.80,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--sim-thresh", type=float, default=0.85,
        help="Cosine similarity gate for track assignment.",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=0.90,
        help="EMA weight for gallery embedding updates.",
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Output video frame rate.",
    )
    args = parser.parse_args()

    # ── Pre-flight ───────────────────────────────────────────────────────────
    _check_ffmpeg()
    det_weights, reid_weights = _check_weights()

    seq_path   = Path(args.sequence)
    all_frames = _validate_sequence(seq_path)
    selected   = (
        all_frames[: args.num_frames]
        if args.num_frames is not None and args.num_frames < len(all_frames)
        else all_frames
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sequence  : {seq_path.name}")
    print(f"Frames    : {len(selected)} / {len(all_frames)} total")
    print(f"Output    : {out_dir.resolve()}")

    # ── Device selection: cuda > mps > cpu ───────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device    : {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models…")
    detector, embed_model = load_models(device, det_weights, reid_weights)
    print("Models loaded.")

    # ── Assemble temp input video from selected frames ────────────────────────
    print(f"Assembling {len(selected)} frames into temp video…")
    tmp_input = _build_temp_video(selected, fps=args.fps)

    # ── Run tracker ───────────────────────────────────────────────────────────
    tmp_raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_raw.close()

    def _progress(frac: float, msg: str) -> None:
        bar = "█" * int(40 * frac) + "░" * (40 - int(40 * frac))
        print(f"\r  [{bar}] {frac * 100:5.1f}%  {msg}", end="", flush=True)

    print("Running tracker…")
    stats = process_video(
        input_path=tmp_input,
        output_path=tmp_raw.name,
        detector=detector,
        embed_model=embed_model,
        device=device,
        det_thresh=args.det_thresh,
        sim_thresh=args.sim_thresh,
        ema_alpha=args.ema_alpha,
        output_fps=args.fps,
        progress_callback=_progress,
    )
    print()  # close the progress bar line

    # ── Extract grid frames from the raw annotated video ──────────────────────
    print("Extracting 6 evenly-spaced frames for grid…")
    grid_frames = _extract_grid_frames(tmp_raw.name, n=6)

    # ── H.264 re-encode → tracked.mp4 ─────────────────────────────────────────
    tracked_path = str(out_dir / "tracked.mp4")
    print("Re-encoding to H.264…")
    _ffmpeg_reencode(tmp_raw.name, tracked_path)

    # ── Frame grid ─────────────────────────────────────────────────────────────
    grid_path = str(out_dir / "frame_grid.png")
    print("Saving frame_grid.png…")
    _save_frame_grid(grid_frames, grid_path)

    # ── Speed stats JSON ───────────────────────────────────────────────────────
    json_path = str(out_dir / "speed_stats.json")
    speed_stats = {
        "sequence":      seq_path.name,
        "device":        str(device),
        "total_frames":  stats["total_frames"],
        "total_seconds": stats["elapsed_seconds"],
        "fps":           stats["avg_fps"],
        "det_thresh":    args.det_thresh,
        "sim_thresh":    args.sim_thresh,
    }
    with open(json_path, "w") as f:
        json.dump(speed_stats, f, indent=2)
    print("Saving speed_stats.json… done")

    # ── Cleanup temp files ─────────────────────────────────────────────────────
    for p in (tmp_input, tmp_raw.name):
        try:
            os.unlink(p)
        except OSError:
            pass

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\nOutputs written to {out_dir.resolve()}/")
    print(f"  tracked.mp4       ({os.path.getsize(tracked_path) // 1024} KB)")
    print(f"  frame_grid.png    ({os.path.getsize(grid_path) // 1024} KB)")
    print(f"  speed_stats.json")
    print(f"\n  Frames processed : {stats['total_frames']}")
    print(f"  Unique IDs       : {stats['unique_ids']}")
    print(f"  Avg FPS          : {stats['avg_fps']}")
    print(f"  Total time       : {stats['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

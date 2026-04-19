"""
app.py — Streamlit web UI for MOT16 Multi-Object Tracker.

Run with:
    streamlit run app.py

Upload a video → configure tracking parameters → click "Run Tracking"
→ view & download annotated output.
"""

import os
import subprocess
import tempfile
import streamlit as st
import torch

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="TrackVision — Multi-Object Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme with gradient accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default header & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- Dark background ---- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ---- Hero banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    animation: shimmer 8s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(5%, 5%) rotate(2deg); }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ---- Glass card ---- */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(102, 126, 234, 0.3);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ---- Metric cards ---- */
.metric-container {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 160px;
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.metric-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
    font-weight: 600;
}

/* ---- Upload area tweaks ---- */
[data-testid="stFileUploader"] {
    border-radius: 14px;
}
[data-testid="stFileUploader"] > div {
    border-radius: 14px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
}

/* ---- Download button ---- */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00c9ff 0%, #92fe9d 100%) !important;
    color: #1a1a2e !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0, 201, 255, 0.3) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 201, 255, 0.5) !important;
}

/* ---- Progress bar ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
    border-radius: 8px !important;
}

/* ---- Video container ---- */
video {
    border-radius: 14px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4) !important;
}

/* ---- Info/Warning/Success boxes ---- */
.stAlert {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🎯 TrackVision</p>
    <p class="hero-subtitle">
        AI-Powered Multi-Object Tracking · Upload a video clip and watch pedestrians
        get detected, identified, and tracked in real time using Faster R-CNN &amp;
        Siamese Re-ID.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🔍 Detection")
    det_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Minimum detector confidence to keep a bounding box.",
    )

    st.markdown("### 🔗 Tracking")
    sim_thresh = st.slider(
        "Similarity Threshold",
        min_value=0.50, max_value=0.99, value=0.85, step=0.01,
        help="Minimum cosine similarity to match a detection to an existing track.",
    )
    ema_alpha = st.slider(
        "EMA Smoothing (α)",
        min_value=0.50, max_value=0.99, value=0.90, step=0.01,
        help="Exponential moving average weight for gallery embedding updates.",
    )
    max_age = st.slider(
        "Max Track Age (frames)",
        min_value=5, max_value=120, value=30, step=5,
        help="Frames a track can go unmatched before being permanently deleted. "
             "Higher = tolerates longer occlusions; lower = faster cleanup.",
    )
    min_hits = st.slider(
        "Min Hits to Confirm",
        min_value=1, max_value=10, value=3, step=1,
        help="Detections must be matched this many frames before the ID is drawn. "
             "Filters spurious one-frame false-positive detections.",
    )

    st.markdown("### 🎬 Output")
    output_fps = st.slider(
        "Output FPS",
        min_value=10, max_value=60, value=25, step=1,
        help="Frame rate of the generated output video.",
    )

    st.markdown("---")

    # Device info
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        st.success(f"🟢 GPU: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        st.success("🟢 Apple MPS (GPU)")
    else:
        st.warning("🟡 CPU mode — processing will be slower")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:rgba(255,255,255,0.35); "
        "font-size:0.8rem;'>TrackVision v1.0<br>Powered by PyTorch &amp; "
        "Streamlit</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Upload Section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="glass-card">
    <div class="section-header">📤 Upload Your Video</div>
    <p style="color: rgba(255,255,255,0.55); font-size: 0.95rem; margin-bottom: 12px;">
        Drag and drop or browse for a video file.
        Supported formats: <strong>MP4, AVI, MOV, MKV</strong>
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------------
# Processing Section
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    # Preview the uploaded video
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">👁️ Input Preview</div>
    </div>
    """, unsafe_allow_html=True)
    st.video(uploaded_file)

    st.markdown("")  # spacer

    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_clicked = st.button(
            "🚀  Run Tracking",
            use_container_width=True,
        )

    if run_clicked:
        # Determine device
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Save uploaded file to temp location
        tmp_input = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        )
        tmp_input.write(uploaded_file.getvalue())
        tmp_input.close()

        tmp_output = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        )
        tmp_output.close()

        try:
            # ---- Load models ----
            with st.spinner("🔧 Loading detection & Re-ID models…"):
                from tracker_engine import load_models, process_video
                detector, embed_model = load_models(device)

            st.success("✅ Models loaded successfully!")

            # ---- Process video ----
            st.markdown("""
            <div class="glass-card">
                <div class="section-header">⏳ Processing</div>
            </div>
            """, unsafe_allow_html=True)

            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress_cb(frac: float, msg: str):
                progress_bar.progress(frac)
                status_text.markdown(
                    f"<p style='color:rgba(255,255,255,0.6); "
                    f"font-size:0.9rem;'>{msg}</p>",
                    unsafe_allow_html=True,
                )

            stats = process_video(
                input_path=tmp_input.name,
                output_path=tmp_output.name,
                detector=detector,
                embed_model=embed_model,
                device=device,
                det_thresh=det_thresh,
                sim_thresh=sim_thresh,
                ema_alpha=ema_alpha,
                max_age=max_age,
                min_hits=min_hits,
                output_fps=output_fps,
                progress_callback=_progress_cb,
            )

            progress_bar.progress(1.0)
            status_text.markdown(
                "<p style='color:#92fe9d; font-size:0.95rem; font-weight:600;'>"
                "✅ Processing complete!</p>",
                unsafe_allow_html=True,
            )

            # ---- Results Section ----
            st.markdown("---")
            st.markdown("""
            <div class="glass-card">
                <div class="section-header">📊 Results</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <p class="metric-value">{stats['total_frames']}</p>
                    <p class="metric-label">Total Frames</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['unique_ids']}</p>
                    <p class="metric-label">Unique IDs (Total)</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['active_ids']}</p>
                    <p class="metric-label">Active IDs (End)</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['elapsed_seconds']}s</p>
                    <p class="metric-label">Processing Time</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['avg_fps']}</p>
                    <p class="metric-label">Avg FPS</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Output video
            st.markdown("""
            <div class="glass-card">
                <div class="section-header">🎬 Tracked Output</div>
            </div>
            """, unsafe_allow_html=True)

            # Re-encode to H.264 for browser compatibility
            tmp_h264 = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            )
            tmp_h264.close()

            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_output.name,
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    tmp_h264.name,
                ],
                capture_output=True, text=True,
            )

            # Use H.264 version if ffmpeg succeeded, else fall back
            if ffmpeg_result.returncode == 0:
                display_path = tmp_h264.name
            else:
                display_path = tmp_output.name
                st.warning(
                    "⚠️ Could not re-encode video for browser playback. "
                    "Use the download button below to view the result."
                )

            with open(display_path, "rb") as vf:
                video_bytes = vf.read()

            st.video(video_bytes)

            st.markdown("<br>", unsafe_allow_html=True)

            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="⬇️  Download Tracked Video",
                    data=video_bytes,
                    file_name="tracked_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Clean up temp files
            for path in [tmp_input.name, tmp_output.name]:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            try:
                os.unlink(tmp_h264.name)
            except (OSError, NameError):
                pass

else:
    # Placeholder when no file is uploaded
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px 32px;">
        <p style="font-size: 3rem; margin-bottom: 12px;">🎥</p>
        <p style="color: rgba(255,255,255,0.5); font-size: 1.1rem; font-weight: 500;">
            Upload a video above to get started
        </p>
        <p style="color: rgba(255,255,255,0.3); font-size: 0.9rem;">
            The tracker will detect and follow pedestrians across frames using
            deep learning
        </p>
    </div>
    """, unsafe_allow_html=True)

"""
🎬 Hybrid Auto-Dubbing Pipeline (SRT Sync)
Hugging Face Space Frontend Application

Architecture:
- Frontend (Hugging Face Space CPU tier): Handles UI, SRT subtitle parsing (pysrt),
  round-robin distributed API routing across Google Colab GPU backends,
  and millisecond-accurate timeline stitching using Pydub.
- Backend (Google Colab T4 GPU Cluster): Executes the adapted Bhojpuri XTTS-v2
  model inference and returns individual dialogue WAV chunks.
"""

from datetime import datetime
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr
from gradio_client import Client
try:
    from gradio_client import handle_file
except ImportError:
    def handle_file(path):
        return str(path)

from pydub import AudioSegment
import pysrt

# Base Paths & Fixed Reference Voice
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_VOICE_PATH = BASE_DIR / "reference_voice.wav"
OUTPUT_AUDIO_PATH = BASE_DIR / "final_dubbed_output.wav"

# Global client cache to reuse Gradio Client HTTP sessions
CLIENT_CACHE: Dict[str, Client] = {}

CUSTOM_CSS = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    color: white;
}
.header-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #c7d2fe;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}
"""


def check_reference_voice() -> str:
    """Verifies that reference_voice.wav is present in the root directory."""
    if REFERENCE_VOICE_PATH.is_file():
        size_mb = REFERENCE_VOICE_PATH.stat().st_size / (1024 * 1024)
        return f"✅ `reference_voice.wav` detected in root folder ({size_mb:.2f} MB)"
    return "⚠️ `reference_voice.wav` NOT FOUND in root directory. Please place it in the root before dubbing."


def get_or_create_client(url: str) -> Client:
    """Retrieves an existing Gradio Client or establishes a new connection."""
    clean_url = url.strip().rstrip("/")
    if clean_url not in CLIENT_CACHE:
        CLIENT_CACHE[clean_url] = Client(clean_url)
    return CLIENT_CACHE[clean_url]


def run_hybrid_dubbing(
    colab_url_1: str,
    colab_url_2: str,
    colab_url_3: str,
    srt_file_path: Optional[str],
    target_language: str,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[Optional[str], str]:
    """
    Core Logic Function:
    1. Ensures reference_voice.wav is used dynamically from the root folder.
    2. Parses uploaded SRT using pysrt.
    3. Calculates total duration (end time of last subtitle in ms + 2000ms buffer).
       Creates a silent AudioSegment canvas using pydub.
    4. Iterates through each subtitle chunk, extracting text and start time in ms.
       Passes the text EXACTLY as it is in the SRT (no alteration or cleaning).
    5. Implements Round-Robin URL selection across provided Colab URLs.
    6. Uses gradio_client.Client to dispatch requests with network timeout/retry logic.
    7. Uses pydub.AudioSegment.overlay() to place returned audio onto the silent canvas
       at the exact start time.
    8. Exports as final_dubbed_output.wav and returns it to the UI.
    """
    logs: List[str] = []

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        logs.append(entry)
        print(entry)

    start_time = time.time()
    log("🚀 Starting Hybrid Anime Dubbing Pipeline...")

    # 1. Verify reference_voice.wav exists in the local root directory
    if not REFERENCE_VOICE_PATH.is_file():
        err_msg = (
            "FATAL ERROR: 'reference_voice.wav' not found in root directory! "
            "Please upload/place your reference voice audio file named 'reference_voice.wav' "
            "in the root directory of this project."
        )
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🎙️ Using root reference voice: `{REFERENCE_VOICE_PATH.name}` ({REFERENCE_VOICE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Validate & Filter Colab API URLs
    raw_urls = [colab_url_1, colab_url_2, colab_url_3]
    active_urls = [u.strip().rstrip("/") for u in raw_urls if u and u.strip()]

    if not active_urls:
        err_msg = "Google Colab API URL 1 is mandatory! Please paste your running Colab public link."
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🌐 Active Colab GPU Backends ({len(active_urls)} connected):")
    for idx, u in enumerate(active_urls, start=1):
        log(f"   • Endpoint #{idx}: {u}")

    # 3. Parse SRT Subtitle File using pysrt
    if not srt_file_path or not os.path.isfile(srt_file_path):
        err_msg = "Please upload an .srt subtitle file!"
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"📄 Parsing subtitle file: `{Path(srt_file_path).name}` with pysrt...")
    try:
        try:
            subs = pysrt.open(srt_file_path, encoding="utf-8")
        except Exception:
            subs = pysrt.open(srt_file_path, encoding="latin-1")
    except Exception as e:
        err_msg = f"Failed to parse SRT file: {e}"
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    if not subs:
        err_msg = "The uploaded SRT file contains no subtitle entries."
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    num_subs = len(subs)
    log(f"✅ Successfully loaded {num_subs} subtitle dialogue blocks.")

    # 4. Calculate Total Duration & Create Silent Canvas
    # End time of last subtitle in ms + 2000ms buffer
    last_sub = subs[-1]
    last_sub_end_ms = last_sub.end.ordinal
    total_duration_ms = last_sub_end_ms + 2000

    log(f"⏱️ Master Timeline: 0 ms -> {last_sub_end_ms} ms (Canvas with 2000ms buffer: {total_duration_ms} ms / {total_duration_ms/1000.0:.2f}s)")
    log("🎼 Initializing silent audio canvas with Pydub (24,000 Hz, Mono)...")
    canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)

    # 5. Iterate Through Subtitle Chunks with Round-Robin Load Balancing
    log(f"🎬 Dubbing dialogue into target language: **{target_language.upper()}**...")
    ref_param = handle_file(str(REFERENCE_VOICE_PATH))
    successful_chunks = 0

    for i, sub in enumerate(subs):
        chunk_idx = i + 1
        start_time_ms = sub.start.ordinal
        end_time_ms = sub.end.ordinal

        # Pass text EXACTLY as it is in the SRT (no altering or cleaning)
        chunk_text = sub.text

        if not chunk_text or not chunk_text.strip():
            log(f"⏩ Subtitle #{chunk_idx} is empty. Skipping.")
            continue

        progress_val = i / num_subs
        progress(progress_val, desc=f"Dubbing Chunk {chunk_idx}/{num_subs} [{target_language.upper()}]...")

        # Round-Robin URL Selection: cycle through provided Colab URLs
        assigned_worker_idx = i % len(active_urls)
        target_url = active_urls[assigned_worker_idx]

        log(f"⚡ Subtitle #{chunk_idx} [{start_time_ms}ms -> {end_time_ms}ms]: '{chunk_text[:35]}...' -> Worker #{assigned_worker_idx + 1}")

        # Dispatch with Retry & Timeout Handling
        returned_audio_path = None
        attempt_urls = [target_url] + [u for u in active_urls if u != target_url]

        for attempt_num, worker_url in enumerate(attempt_urls, start=1):
            try:
                client = get_or_create_client(worker_url)
                # Call Colab Gradio API (/predict)
                result = client.predict(
                    text_chunk=chunk_text,  # EXACT raw text from SRT
                    target_lang=target_language.strip().lower(),
                    reference_voice=ref_param,
                    api_name="/predict",
                )
                returned_audio_path = result
                break  # Successful response
            except Exception as e:
                log(f"⚠️ [Attempt {attempt_num}] Worker ({worker_url}) timed out or failed on Chunk #{chunk_idx}: {e}")
                time.sleep(1)

        if not returned_audio_path or not os.path.exists(returned_audio_path):
            log(f"❌ Failed to synthesize Chunk #{chunk_idx} after retrying across available endpoints. Skipping.")
            continue

        # 6. Overlay Audio Chunk onto Silent Canvas at exact start_time_ms
        try:
            chunk_audio = AudioSegment.from_file(returned_audio_path)
            # Ensure sample rate and channel alignment with canvas
            if chunk_audio.frame_rate != canvas.frame_rate:
                chunk_audio = chunk_audio.set_frame_rate(canvas.frame_rate)
            if chunk_audio.channels != canvas.channels:
                chunk_audio = chunk_audio.set_channels(canvas.channels)

            # Smooth edge fading to eliminate boundary clicks
            if len(chunk_audio) > 40:
                chunk_audio = chunk_audio.fade_in(15).fade_out(15)

            # Overlay chunk onto master canvas at exact subtitle start time
            canvas = canvas.overlay(chunk_audio, position=start_time_ms)
            successful_chunks += 1
            log(f"✅ Chunk #{chunk_idx} stitched at {start_time_ms}ms ({len(chunk_audio)}ms duration)")

        except Exception as e:
            log(f"❌ Error overlaying audio for Chunk #{chunk_idx}: {e}")

    # 7. Export Final Dubbed Output
    progress(0.96, desc="Exporting master audio track...")
    log("✂️ Exporting stitched composite audio to final_dubbed_output.wav...")

    try:
        canvas.export(str(OUTPUT_AUDIO_PATH), format="wav")
        elapsed = time.time() - start_time
        log(f"🎉 Dubbing pipeline finished in {elapsed:.2f}s!")
        log(f"📊 Successfully synthesized & stitched {successful_chunks}/{num_subs} dialogue blocks.")
        log(f"📁 Output file exported: `{OUTPUT_AUDIO_PATH.name}` ({OUTPUT_AUDIO_PATH.stat().st_size / (1024*1024):.2f} MB)")
        progress(1.0, desc="Dubbing complete!")
        return str(OUTPUT_AUDIO_PATH), "\n".join(logs)
    except Exception as e:
        err_msg = f"Failed to export final master WAV: {e}"
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)


def build_app() -> gr.Blocks:
    """Builds the Gradio Blocks User Interface."""
    with gr.Blocks(title="Hybrid Auto-Dubbing Pipeline (SRT Sync)") as demo:
        # Header Banner
        with gr.Row():
            gr.HTML(
                """
                <div class="header-box">
                    <div class="header-title">🎬 Hybrid Auto-Dubbing Pipeline (SRT Sync)</div>
                    <div style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 12px;">
                        Enterprise-Grade Anime & Video Dubbing using <b>Bhojpuri-Adapted Coqui XTTS-v2</b>, 
                        <b>Hugging Face Space Frontend</b>, and <b>Distributed Google Colab GPU Backends</b>.
                    </div>
                    <div>
                        <span class="badge">🧠 Bhojpuri-Adapted XTTS-v2</span>
                        <span class="badge">⚖️ Round-Robin Multi-GPU Load Balancing</span>
                        <span class="badge">⏱️ pysrt Millisecond Lip-Sync</span>
                        <span class="badge">✂️ Pydub Timeline Stitching</span>
                        <span class="badge">🎙️ Hardcoded Root Reference Voice</span>
                    </div>
                </div>
                """
            )

        with gr.Row():
            # LEFT COLUMN: Inputs & API Configuration
            with gr.Column(scale=5):
                gr.Markdown("### ⚙️ 1. Google Colab GPU API Endpoints")
                gr.Markdown(
                    "Launch the backend notebook in Google Colab (T4 GPU), copy the generated `https://xxxx.gradio.live` link, and paste below."
                )

                colab_url_1 = gr.Textbox(
                    label="Google Colab API URL 1 (Mandatory Primary)",
                    placeholder="https://xxxxxxxx.gradio.live",
                    lines=1,
                )
                colab_url_2 = gr.Textbox(
                    label="Google Colab API URL 2 (Optional Load Balancer)",
                    placeholder="https://yyyyyyyy.gradio.live (Leave blank if running 1 Colab)",
                    lines=1,
                )
                colab_url_3 = gr.Textbox(
                    label="Google Colab API URL 3 (Optional Load Balancer)",
                    placeholder="https://zzzzzzzz.gradio.live (Leave blank if running 1 or 2 Colabs)",
                    lines=1,
                )

                gr.Markdown("### 📄 2. Subtitles & Target Language")
                srt_input = gr.File(
                    label="Upload SRT Subtitle File (.srt)",
                    file_types=[".srt"],
                    type="filepath",
                )

                target_lang = gr.Dropdown(
                    label="Target Language",
                    choices=[
                        ("Hindi (hi)", "hi"),
                        ("French (fr)", "fr"),
                        ("Spanish (es)", "es"),
                        ("Portuguese (pt)", "pt"),
                    ],
                    value="hi",
                    info="Select target speech translation language.",
                )

                # Reference voice status check indicator (No upload button in UI)
                voice_status = gr.Markdown(value=check_reference_voice())

                start_btn = gr.Button(
                    "🎬 Start Dubbing",
                    variant="primary",
                    size="lg",
                )

            # RIGHT COLUMN: Outputs & Live Execution Logs
            with gr.Column(scale=6):
                gr.Markdown("### 🎧 3. Dubbed Audio Master Output")
                audio_output = gr.Audio(
                    label="Dubbed Audio Output (final_dubbed_output.wav)",
                    type="filepath",
                )

                gr.Markdown("### 📜 4. Real-Time Status Logs")
                logs_output = gr.Textbox(
                    label="Status Logs",
                    lines=12,
                    autoscroll=True,
                    placeholder="Pipeline logs will stream here when dubbing starts...",
                )

        # Wire Submit Button Event
        start_btn.click(
            fn=run_hybrid_dubbing,
            inputs=[
                colab_url_1,
                colab_url_2,
                colab_url_3,
                srt_input,
                target_lang,
            ],
            outputs=[
                audio_output,
                logs_output,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    app.launch(share=False)

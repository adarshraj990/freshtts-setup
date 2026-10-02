"""
🎬 Hybrid Auto-Dubbing Pipeline (SRT Sync)
Hugging Face Space Frontend Application with Multi-API Pool Load Balancer

Architecture & Highlights:
- Multi-API Pool: Connects up to 5 distributed Google Colab / Hugging Face Space GPU backends.
- Parallel Chunk Distribution: Distributes subtitle dialogue blocks concurrently across all active backends.
- Time-Aligned Subtitle Sync: Millisecond-accurate timeline placement via pysrt.
- Chronological Audio Stitching: Seamless master canvas overlay using Pydub.
- Root Reference Voice: Automatically utilizes `reference_voice.wav` from the root directory.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

# ── Monkey-patch missing HfFolder for huggingface_hub >= 0.23.0 compatibility ──
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            path_token = None

            @classmethod
            def get_token(cls):
                try:
                    return huggingface_hub.get_token()
                except Exception:
                    return None

            @classmethod
            def save_token(cls, token):
                try:
                    huggingface_hub.login(token=token)
                except Exception:
                    pass

            @classmethod
            def delete_token(cls):
                try:
                    huggingface_hub.logout()
                except Exception:
                    pass

        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# ── Monkey-patch gradio_client schema parsing bug (bool is not iterable) ──────
try:
    import gradio_client.utils
    if hasattr(gradio_client.utils, "get_type"):
        _orig_get_type = gradio_client.utils.get_type
        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            if not isinstance(schema, dict):
                return "any"
            return _orig_get_type(schema)
        gradio_client.utils.get_type = _safe_get_type

    if hasattr(gradio_client.utils, "_json_schema_to_python_type"):
        _orig_schema_to_type = gradio_client.utils._json_schema_to_python_type
        def _safe_schema_to_type(schema, defs):
            if isinstance(schema, bool):
                return "bool"
            try:
                return _orig_schema_to_type(schema, defs)
            except Exception:
                return "Any"
        gradio_client.utils._json_schema_to_python_type = _safe_schema_to_type
except Exception:
    pass

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

# Global client cache to reuse Gradio Client HTTP sessions across threads
CLIENT_CACHE: Dict[str, Client] = {}
CLIENT_CACHE_LOCK = threading.Lock()


def check_reference_voice() -> str:
    """Verifies that reference_voice.wav is present in the root directory."""
    if REFERENCE_VOICE_PATH.is_file():
        size_mb = REFERENCE_VOICE_PATH.stat().st_size / (1024 * 1024)
        return f"✅ `reference_voice.wav` detected in root folder ({size_mb:.2f} MB)"
    return "⚠️ `reference_voice.wav` NOT FOUND in root directory. Please place it in the root before dubbing."


def get_or_create_client(url: str) -> Client:
    """Retrieves an existing Gradio Client or establishes a new connection in a thread-safe manner."""
    clean_url = url.strip().rstrip("/")
    with CLIENT_CACHE_LOCK:
        if clean_url not in CLIENT_CACHE:
            CLIENT_CACHE[clean_url] = Client(clean_url)
        return CLIENT_CACHE[clean_url]


def synthesize_chunk_task(
    chunk_tuple: Tuple[int, int, int, str],
    active_urls: List[str],
    ref_param,
    target_lang_code: str,
) -> Tuple[int, int, Optional[str], Optional[str], str]:
    """
    Synthesizes a single subtitle chunk with round-robin primary assignment and failover retry.
    Returns: (chunk_idx, start_time_ms, returned_audio_path, error_msg, worker_used)
    """
    chunk_idx, start_time_ms, end_time_ms, chunk_text = chunk_tuple

    # Round-robin initial worker assignment
    primary_idx = (chunk_idx - 1) % len(active_urls)
    ordered_urls = [active_urls[primary_idx]] + [u for i, u in enumerate(active_urls) if i != primary_idx]

    last_error = None
    for attempt, worker_url in enumerate(ordered_urls, start=1):
        try:
            client = get_or_create_client(worker_url)
            result = client.predict(
                text_chunk=chunk_text,
                target_lang=target_lang_code,
                reference_voice=ref_param,
                api_name="/predict",
            )
            if result and os.path.exists(result):
                return chunk_idx, start_time_ms, result, None, worker_url
        except Exception as e:
            last_error = f"{worker_url}: {e}"
            time.sleep(0.5)

    return chunk_idx, start_time_ms, None, last_error or "All endpoints failed", "None"


def run_hybrid_dubbing(
    api_url_1: str,
    api_url_2: str,
    api_url_3: str,
    api_url_4: str,
    api_url_5: str,
    srt_file_path: Optional[str],
    target_language: str,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[Optional[str], str]:
    """
    Multi-API Pool Load Balancer:
    1. Collects and validates all active backend API URLs (URLs 1–5).
    2. Verifies the root reference_voice.wav file.
    3. Parses uploaded SRT subtitles using pysrt.
    4. Distributes generation requests across all available backends concurrently.
    5. Reassembles synthesized audio chunks in strictly chronological order.
    6. Stitches onto a Pydub silent canvas and exports final_dubbed_output.wav.
    """
    logs: List[str] = []
    log_lock = threading.Lock()

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        with log_lock:
            logs.append(entry)
        print(entry)

    start_time = time.time()
    log("🚀 Initializing Multi-API Pool Load Balancer for XTTS-v2 Dubbing...")

    # 1. Verify root reference voice
    if not REFERENCE_VOICE_PATH.is_file():
        err_msg = (
            "FATAL ERROR: 'reference_voice.wav' not found in root directory! "
            "Please ensure reference_voice.wav is present in the workspace root."
        )
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🎙️ Active Reference Voice: `{REFERENCE_VOICE_PATH.name}` ({REFERENCE_VOICE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Collect & Validate Active API Pool Endpoints
    raw_urls = [api_url_1, api_url_2, api_url_3, api_url_4, api_url_5]
    active_urls = [u.strip().rstrip("/") for u in raw_urls if u and u.strip()]

    if not active_urls:
        err_msg = "At least one API URL (API URL 1) is mandatory! Please paste your running backend link."
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🌐 Multi-API Pool initialized with {len(active_urls)} active backend(s):")
    for idx, u in enumerate(active_urls, start=1):
        log(f"   • Backend #{idx}: {u}")

    # 3. Parse SRT Subtitles
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

    # Filter non-empty subtitle blocks
    chunks_to_process = []
    for i, sub in enumerate(subs):
        text = sub.text
        if text and text.strip():
            chunks_to_process.append((i + 1, sub.start.ordinal, sub.end.ordinal, text))

    num_chunks = len(chunks_to_process)
    log(f"✅ Loaded {len(subs)} subtitle blocks ({num_chunks} valid dialogue chunks to synthesize).")

    # 4. Canvas Initialization
    last_sub = subs[-1]
    total_duration_ms = last_sub.end.ordinal + 2000
    log(f"⏱️ Master Timeline: 0 ms -> {last_sub.end.ordinal} ms (Canvas with 2000ms buffer: {total_duration_ms} ms)")
    log("🎼 Initializing silent master canvas with Pydub (24,000 Hz, Mono)...")
    canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)

    # 5. Distributed Chunk Generation Across Pool
    target_lang_code = target_language.strip().lower()
    ref_param = handle_file(str(REFERENCE_VOICE_PATH))
    max_workers = min(len(active_urls), 8)

    log(f"🎬 Dubbing into target language: **{target_lang_code.upper()}**")
    log(f"⚡ Dispatching {num_chunks} chunks across {len(active_urls)} endpoint(s) with {max_workers} concurrent thread(s)...")

    chunk_results: List[Tuple[int, int, str]] = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                synthesize_chunk_task,
                chunk,
                active_urls,
                ref_param,
                target_lang_code,
            ): chunk
            for chunk in chunks_to_process
        }

        for future in as_completed(future_to_chunk):
            chunk_idx, start_time_ms, audio_path, err, worker_used = future.result()
            completed_count += 1
            progress_ratio = completed_count / num_chunks
            progress(
                progress_ratio,
                desc=f"Synthesizing [{completed_count}/{num_chunks}] across API pool...",
            )

            if audio_path and os.path.exists(audio_path):
                log(f"✅ Chunk #{chunk_idx} completed by {worker_used}")
                chunk_results.append((chunk_idx, start_time_ms, audio_path))
            else:
                log(f"❌ Chunk #{chunk_idx} failed across available pool: {err}")

    # 6. Assembly: Chronological Stitching onto Master Canvas
    progress(0.95, desc="Stitching audio chunks into master track...")
    chunk_results.sort(key=lambda x: x[0])  # Guarantee exact chronological order
    log(f"🎼 Assembling {len(chunk_results)} generated chunks onto master timeline canvas...")

    successful_chunks = 0
    for chunk_idx, start_time_ms, audio_path in chunk_results:
        try:
            chunk_audio = AudioSegment.from_file(audio_path)
            if chunk_audio.frame_rate != canvas.frame_rate:
                chunk_audio = chunk_audio.set_frame_rate(canvas.frame_rate)
            if chunk_audio.channels != canvas.channels:
                chunk_audio = chunk_audio.set_channels(canvas.channels)
            if len(chunk_audio) > 40:
                chunk_audio = chunk_audio.fade_in(15).fade_out(15)

            canvas = canvas.overlay(chunk_audio, position=start_time_ms)
            successful_chunks += 1
        except Exception as e:
            log(f"⚠️ Error overlaying Chunk #{chunk_idx}: {e}")

    # 7. Master Output Export
    progress(0.98, desc="Exporting master WAV...")
    log("✂️ Exporting stitched composite audio to final_dubbed_output.wav...")

    try:
        canvas.export(str(OUTPUT_AUDIO_PATH), format="wav")
        elapsed = time.time() - start_time
        log(f"🎉 Dubbing pipeline finished in {elapsed:.2f}s!")
        log(f"📊 Successfully synthesized & stitched {successful_chunks}/{num_chunks} dialogue blocks.")
        log(f"📁 Master output ready: `{OUTPUT_AUDIO_PATH.name}` ({OUTPUT_AUDIO_PATH.stat().st_size / (1024*1024):.2f} MB)")
        progress(1.0, desc="Dubbing complete!")
        return str(OUTPUT_AUDIO_PATH), "\n".join(logs)
    except Exception as e:
        err_msg = f"Failed to export final master WAV: {e}"
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)


def build_app() -> gr.Blocks:
    """Builds the polished Gradio Blocks User Interface with Multi-API Pool Load Balancer."""
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="🎬 Hybrid Auto-Dubbing Pipeline (Multi-API Pool)") as demo:
        # Welcoming Header
        gr.Markdown(
            """
            # 🎬 Hybrid Auto-Dubbing Pipeline (SRT Sync)
            ### 🌐 Distributed Multi-API Pool Load Balancer for Coqui XTTS-v2
            Distribute dialogue speech cloning across multiple Google Colab / Hugging Face Space GPU backends with millisecond-accurate subtitle timeline alignment.
            """
        )

        # 1. Multi-API Pool Configuration
        with gr.Group():
            gr.Markdown("### 🌐 1. Multi-API Pool (Distributed XTTS-v2 Endpoints)")
            gr.Markdown(
                "Provide Google Colab (`.gradio.live`) or Hugging Face Space API URLs. "
                "Subtitle chunks are automatically distributed across all active endpoints in parallel for linear speedups."
            )

            api_url_1 = gr.Textbox(
                label="API URL 1 (Mandatory Primary)",
                placeholder="https://xxxxxxxx.gradio.live (Colab or Space endpoint)",
                lines=1,
            )

            with gr.Accordion("⚙️ Additional API Endpoints (Pool Expansion: URLs 2, 3, 4, 5)", open=False):
                gr.Markdown("Add secondary endpoints to scale parallel dubbing speedups (2x–5x).")
                api_url_2 = gr.Textbox(
                    label="API URL 2 (Optional)",
                    placeholder="https://yyyyyyyy.gradio.live",
                    lines=1,
                )
                api_url_3 = gr.Textbox(
                    label="API URL 3 (Optional)",
                    placeholder="https://zzzzzzzz.gradio.live",
                    lines=1,
                )
                api_url_4 = gr.Textbox(
                    label="API URL 4 (Optional)",
                    placeholder="https://wwwwwwww.gradio.live",
                    lines=1,
                )
                api_url_5 = gr.Textbox(
                    label="API URL 5 (Optional)",
                    placeholder="https://vvvvvvvv.gradio.live",
                    lines=1,
                )

        # 2. Main Two-Column Workflow
        with gr.Row():
            # LEFT COLUMN: User Inputs
            with gr.Column(scale=5):
                gr.Markdown("### 📥 2. Subtitles & Language Configuration")
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
            with gr.Column(scale=5):
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
                api_url_1,
                api_url_2,
                api_url_3,
                api_url_4,
                api_url_5,
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
    try:
        app.launch(server_name="0.0.0.0", server_port=7860)
    except ValueError:
        app.launch()

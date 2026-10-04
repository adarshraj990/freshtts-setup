"""
🎬 High-Throughput Distributed SRT Voice Cloning & Auto-Dubbing Studio
Frontend: Hugging Face Space (Lightweight Gradio Client, Zero GPU Inference)
Backend: Remote Dual-GPU Kaggle / Colab XTTS-v2 Engine (2x T4 GPUs)

Optimizations:
1. Saturated Dual-GPU Parallelism: 5 concurrent threads via ThreadPoolExecutor.
2. In-Memory & Hash-Based Cache: Idempotent caching skips re-synthesizing repeated lines.
3. Subtitle Cleansing: Automatic filtering of empty lines, timestamps, and non-spoken artifacts.
4. Fast In-Memory Audio Assembly: Direct BytesIO processing + pydub concatenation with 200ms gap.
5. Automatic 1-Time Retry & Timeout Guard: Resilient against intermittent dropped network packets.
6. Localtunnel & Ngrok Bypass: Automatic headers bypass tunnel reminder landing pages.
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout/stderr for Unicode safety across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def safe_print(*args, **kwargs):
    """Safely prints to stdout without raising UnicodeEncodeError on cp1252/ascii terminals."""
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            safe_args = [str(a).encode("ascii", "replace").decode("ascii") for a in args]
            print(*safe_args, **kwargs)
        except Exception:
            pass


# ── Monkey-patch missing HfFolder for huggingface_hub compatibility ──────────
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
                pass
            @classmethod
            def delete_token(cls):
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
        def _safe_schema_to_type(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            if not isinstance(schema, dict):
                return "Any"
            try:
                return _orig_schema_to_type(schema, defs)
            except Exception:
                return "Any"
        gradio_client.utils._json_schema_to_python_type = _safe_schema_to_type
except Exception:
    pass

# ── Monkey-patch gradio get_api_info for robust ASGI route initialization ─────
try:
    import gradio.blocks
    if hasattr(gradio.blocks.Block, "get_api_info"):
        _orig_get_api_info = gradio.blocks.Block.get_api_info
        def _safe_get_api_info(self):
            try:
                return _orig_get_api_info(self)
            except Exception:
                return {}
        gradio.blocks.Block.get_api_info = _safe_get_api_info
except Exception:
    pass

import gradio as gr
from pydub import AudioSegment
import pysrt
import requests

# Base Paths & Directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_VOICE = str(BASE_DIR / "reference_voice.wav")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard Localtunnel & Ngrok bypass headers
TUNNEL_HEADERS = {
    "Bypass-Tunnel-Reminder": "true",
    "bypass-tunnel-reminder": "true",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "AudioGenFlow-DualGPU-Client/1.0",
}

# In-Memory Cache for Idempotent Operations (Thread-Safe)
AUDIO_CACHE: Dict[str, bytes] = {}
CACHE_LOCK = threading.Lock()


# ── Subtitle Cleansing & Filtering ────────────────────────────────────────────

def clean_subtitle_text(raw_text: str) -> str:
    """
    Strips formatting tags, non-spoken subtitle markers, and whitespace artifacts.
    Prevents sending empty lines, timestamps, or pure music notes over the network.
    """
    if not raw_text:
        return ""
    # Strip HTML tags (e.g. <i>, <b>, <font color="...">)
    text = re.sub(r"<[^>]+>", "", raw_text)
    # Strip ASS/SSA tags (e.g. {\an8})
    text = re.sub(r"\{[^}]+\}", "", text)
    # Strip sound effect annotations: [Music], (Applause), ♪, etc.
    text = re.sub(r"\[.*?\]|\(.*?\)|♪|♫|#", "", text)
    # Normalize whitespaces and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_cache_key(text: str, language: str, voice_hash: str) -> str:
    """Computes a unique MD5 hash key for caching speech audio."""
    raw = f"{text}|{language}|{voice_hash}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ── Single-Chunk Synthesis with 1-Time Retry ───────────────────────────────────

def synthesize_chunk_with_retry(
    endpoint: str,
    text: str,
    language: str,
    ref_audio_path: str,
    mime_type: str,
    timeout_sec: int = 120,
    max_retries: int = 1,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Dispatches a single subtitle chunk to the Kaggle Dual-GPU backend.
    Includes automated 1-time retry for intermittent dropped packets.
    """
    data = {
        "text": text,
        "text_chunk": text,
        "language": language,
        "target_lang": language,
    }

    filename = Path(ref_audio_path).name

    for attempt in range(max_retries + 1):
        try:
            with open(ref_audio_path, "rb") as f1, open(ref_audio_path, "rb") as f2:
                files = [
                    ("speaker_wav", (filename, f1, mime_type)),
                    ("reference_voice", (filename, f2, mime_type)),
                ]
                resp = requests.post(
                    endpoint,
                    headers=TUNNEL_HEADERS,
                    data=data,
                    files=files,
                    timeout=timeout_sec,
                )

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "").lower()

                # Handle JSON response containing base64 audio
                if "application/json" in content_type:
                    try:
                        payload = resp.json()
                        b64_audio = payload.get("audio_base64") or payload.get("audio")
                        if b64_audio:
                            return base64.b64decode(b64_audio), None
                    except Exception as json_err:
                        return None, f"JSON parse error: {json_err}"

                # Direct binary audio stream
                return resp.content, None

            elif attempt < max_retries and resp.status_code in [502, 503, 504]:
                time.sleep(1.0)
                continue
            else:
                return None, f"HTTP {resp.status_code}: {resp.text[:120]}"

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            return None, f"Network Timeout / Disconnect: {type(net_err).__name__}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    return None, "Failed after retry"


# ── High-Throughput Saturated SRT Voice Cloning Engine ────────────────────────

def run_high_throughput_srt_pipeline(
    kaggle_url: str,
    srt_file: Optional[str],
    language_code: str,
    use_default_voice: bool,
    custom_voice_file: Optional[str],
    concurrent_workers: int = 5,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Max-Throughput Dual-GPU SRT Dubbing Engine:
    1. Validates dynamic URL and sanitizes endpoint.
    2. Resolves reference voice sample and calculates voice MD5 hash.
    3. Parses and cleans SRT subtitle blocks (skips empty lines/artifacts).
    4. Saturated Parallelism: 5 concurrent threads via ThreadPoolExecutor.
    5. In-Memory Cache Lookup: Skips re-synthesizing repeated dialogue lines.
    6. Fast In-Memory Assembly: Direct BytesIO concatenation with 200ms gap.
    7. Memory/Disk Optimization: Discards temporary chunk buffers immediately.
    """
    logs: List[str] = []
    log_lock = threading.Lock()

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        with log_lock:
            logs.append(entry)
        safe_print(entry)

    t_start = time.time()
    log("🚀 Initializing Saturated Dual-GPU SRT Dubbing Pipeline...")

    # 1. Validate & Sanitize Kaggle API URL
    if not kaggle_url or not kaggle_url.strip():
        err = "Please enter your active Kaggle API URL (Localtunnel or Ngrok)!"
        log(f"❌ {err}")
        raise gr.Error(err)

    clean_url = kaggle_url.strip().rstrip("/")
    if not clean_url.endswith("/voice_clone_synthesis"):
        endpoint = f"{clean_url}/voice_clone_synthesis"
    else:
        endpoint = clean_url

    log(f"🌐 Backend Target Endpoint: `{endpoint}`")

    # 2. Resolve Reference Voice Audio & Hash
    if use_default_voice:
        if os.path.isfile(DEFAULT_REFERENCE_VOICE):
            ref_path = DEFAULT_REFERENCE_VOICE
            log(f"🎙️ Using Default Voice: `{Path(ref_path).name}` ({os.path.getsize(ref_path)/(1024*1024):.2f} MB)")
        else:
            err = "Built-in reference_voice.wav was not found in the root directory!"
            log(f"❌ {err}")
            raise gr.Error(err)
    else:
        if not custom_voice_file or not os.path.isfile(custom_voice_file):
            err = "Please upload a reference voice sample or tick 'Use Default Voice (reference_voice.wav)'!"
            log(f"❌ {err}")
            raise gr.Error(err)
        ref_path = custom_voice_file
        log(f"🎙️ Using Custom Voice: `{Path(ref_path).name}` ({os.path.getsize(ref_path)/(1024*1024):.2f} MB)")

    # Compute voice file MD5 for cache invalidation
    with open(ref_path, "rb") as vf:
        voice_hash = hashlib.md5(vf.read(1024 * 512)).hexdigest()

    mime_type = "audio/mpeg" if Path(ref_path).name.lower().endswith(".mp3") else "audio/wav"

    # 3. Parse and Clean SRT Subtitles
    if not srt_file or not os.path.isfile(srt_file):
        err = "Please upload a valid .srt subtitle file!"
        log(f"❌ {err}")
        raise gr.Error(err)

    log(f"📄 Parsing and cleansing SRT subtitles: `{Path(srt_file).name}`...")
    try:
        try:
            subs = pysrt.open(srt_file, encoding="utf-8")
        except Exception:
            subs = pysrt.open(srt_file, encoding="latin-1")
    except Exception as e:
        err = f"Failed to parse SRT file: {e}"
        log(f"❌ {err}")
        raise gr.Error(err)

    if not subs:
        err = "The uploaded SRT file contains no subtitle entries."
        log(f"❌ {err}")
        raise gr.Error(err)

    # Filter out empty lines, subtitle tags, and pure timestamp artifacts
    cleaned_chunks: List[Tuple[int, int, str]] = []
    skipped_count = 0

    for i, sub in enumerate(subs, start=1):
        cleaned_text = clean_subtitle_text(sub.text)
        # Check if text contains spoken characters (alphanumeric in any language)
        if cleaned_text and re.search(r"\w", cleaned_text, re.UNICODE):
            cleaned_chunks.append((i, sub.start.ordinal, cleaned_text))
        else:
            skipped_count += 1

    total_chunks = len(cleaned_chunks)
    log(f"📊 SRT Parsing Complete: {len(subs)} total blocks loaded.")
    if skipped_count > 0:
        log(f"⚡ Skipped {skipped_count} empty / non-spoken artifact blocks to save GPU cycles.")
    log(f"🎯 Valid Dialogue Chunks to Synthesize: **{total_chunks}**")

    if total_chunks == 0:
        err = "No valid spoken dialogue found in the SRT file after cleansing."
        log(f"❌ {err}")
        raise gr.Error(err)

    # 4. Saturated Parallelism Execution (ThreadPoolExecutor)
    num_workers = max(2, min(int(concurrent_workers), 8))
    log(f"⚡ Launching Saturated Dual-GPU Worker Pool with **{num_workers} parallel threads**...")

    generated_segments: Dict[int, AudioSegment] = {}
    completed_chunks = 0
    cache_hits = 0
    lang = language_code.strip().lower()

    def process_sub_block(chunk_info: Tuple[int, int, str]) -> Tuple[int, Optional[AudioSegment], Optional[str]]:
        nonlocal cache_hits
        idx, start_ms, text = chunk_info
        cache_key = get_cache_key(text, lang, voice_hash)

        # 1. Check in-memory cache (Idempotent operation)
        with CACHE_LOCK:
            if cache_key in AUDIO_CACHE:
                cached_bytes = AUDIO_CACHE[cache_key]
                seg = AudioSegment.from_file(io.BytesIO(cached_bytes))
                return idx, seg, "CACHE_HIT"

        # 2. Remote Dual-GPU Synthesis Call with retry
        audio_bytes, err = synthesize_chunk_with_retry(
            endpoint=endpoint,
            text=text,
            language=lang,
            ref_audio_path=ref_path,
            mime_type=mime_type,
            timeout_sec=120,
            max_retries=1,
        )

        if audio_bytes and len(audio_bytes) > 100:
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                # Store in cache
                with CACHE_LOCK:
                    AUDIO_CACHE[cache_key] = audio_bytes
                return idx, seg, None
            except Exception as dec_err:
                return idx, None, f"Audio decode error: {dec_err}"
        else:
            return idx, None, err or "Empty audio response"

    # Dispatch tasks across ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_sub_block, chunk): chunk for chunk in cleaned_chunks
        }

        for future in as_completed(future_to_chunk):
            sub_idx, seg, status = future.result()
            completed_chunks += 1
            progress_ratio = completed_chunks / total_chunks
            elapsed_current = max(0.1, time.time() - t_start)
            throughput = completed_chunks / elapsed_current

            progress(
                progress_ratio,
                desc=f"Synthesizing [{completed_chunks}/{total_chunks}] ({throughput:.1f} chunks/s)...",
            )

            if seg is not None:
                generated_segments[sub_idx] = seg
                if status == "CACHE_HIT":
                    cache_hits += 1
                    log(f"⚡ [Chunk #{sub_idx}/{total_chunks}] Cache Hit (0ms) | Throughput: {throughput:.2f} chunks/s")
                else:
                    log(f"✅ [Chunk #{sub_idx}/{total_chunks}] Generated & Received | Throughput: {throughput:.2f} chunks/s")
            else:
                log(f"⚠️ [Chunk #{sub_idx}/{total_chunks}] Failed: {status}")

    # 5. Fast Audio Assembly with Low-Latency Gap (200ms)
    progress(0.95, desc="Concatenating audio segments in chronological order...")
    log(f"🎼 Assembling {len(generated_segments)} audio segments in exact subtitle index order...")

    if not generated_segments:
        err = "All subtitle segments failed to synthesize. Please check your Kaggle GPU notebook logs."
        log(f"❌ {err}")
        raise gr.Error(err)

    # Sort segments strictly by subtitle index
    sorted_indices = sorted(generated_segments.keys())
    gap_duration_ms = 200
    gap = AudioSegment.silent(duration=gap_duration_ms, frame_rate=24000)

    master_audio = None
    for idx in sorted_indices:
        seg = generated_segments[idx]
        if seg.frame_rate != 24000:
            seg = seg.set_frame_rate(24000)
        if seg.channels != 1:
            seg = seg.set_channels(1)

        # Smooth 15ms micro-fade to avoid boundary clicks
        if len(seg) > 30:
            seg = seg.fade_in(15).fade_out(15)

        if master_audio is None:
            master_audio = seg
        else:
            master_audio = master_audio + gap + seg

    # 6. Direct Export & Immediate Memory Cleanup
    progress(0.98, desc="Exporting optimized master WAV...")
    output_filename = f"dubbed_master_{lang}_{int(time.time())}.wav"
    output_path = OUTPUT_DIR / output_filename

    master_audio.export(str(output_path), format="wav")
    final_file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Free memory buffers immediately
    del generated_segments
    del master_audio

    total_time = time.time() - t_start
    avg_speed = total_chunks / max(0.1, total_time)

    log(f"🎉 Pipeline finished in {total_time:.2f}s ({avg_speed:.2f} dialogue chunks/sec)!")
    log(f"📊 Completed: {completed_chunks}/{total_chunks} chunks ({cache_hits} cache hits).")
    log(f"📁 Master Output File: `{output_filename}` ({final_file_size_mb:.2f} MB)")
    progress(1.0, desc="Dubbing complete!")

    return str(output_path), str(output_path), "\n".join(logs)


# ── Gradio Web UI Layout ──────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="🎬 Saturated Dual-GPU SRT Dubbing Studio") as demo:
        # Header Banner
        gr.Markdown(
            """
            # 🎬 Saturated Dual-GPU SRT Voice Cloning Studio
            ### ⚡ High-Throughput Subtitle Auto-Dubbing via Remote Kaggle Dual T4 GPUs
            Saturate both GPU workers with 5 concurrent threads, in-memory caching, and automated packet-drop retries.
            """
        )

        with gr.Row():
            # LEFT COLUMN: User Inputs
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Connection & Subtitle Configuration")

                kaggle_url_input = gr.Textbox(
                    label="Kaggle API URL (Dynamic Localtunnel / Ngrok)",
                    placeholder="https://famous-sheep-wash.loca.lt or https://xxxx.ngrok-free.app",
                    value="",
                    lines=1,
                    info="Paste your active Kaggle GPU tunnel URL. Automatically routes to /voice_clone_synthesis.",
                )

                srt_file_input = gr.File(
                    label="Upload SRT Subtitle File (.srt)",
                    file_types=[".srt"],
                    type="filepath",
                )

                language_dropdown = gr.Dropdown(
                    label="Target Language Code",
                    choices=[
                        ("Hindi (hi)", "hi"),
                        ("English (en)", "en"),
                        ("Spanish (es)", "es"),
                        ("French (fr)", "fr"),
                        ("German (de)", "de"),
                        ("Italian (it)", "it"),
                        ("Portuguese (pt)", "pt"),
                        ("Russian (ru)", "ru"),
                        ("Arabic (ar)", "ar"),
                        ("Japanese (ja)", "ja"),
                        ("Korean (ko)", "ko"),
                    ],
                    value="hi",
                    info="Language for XTTS-v2 voice synthesis.",
                )

                with gr.Group():
                    use_default_voice_cb = gr.Checkbox(
                        label="🎯 Use Default Voice (reference_voice.wav)",
                        value=True,
                        info="When ticked, automatically uses the root reference_voice.wav file.",
                    )

                    custom_voice_input = gr.Audio(
                        label="Custom Reference Voice (Upload or Record)",
                        type="filepath",
                        value=DEFAULT_REFERENCE_VOICE if os.path.isfile(DEFAULT_REFERENCE_VOICE) else None,
                    )

                def on_toggle_voice(is_checked: bool):
                    if is_checked:
                        val = DEFAULT_REFERENCE_VOICE if os.path.isfile(DEFAULT_REFERENCE_VOICE) else None
                        return gr.update(value=val, interactive=False)
                    else:
                        return gr.update(value=None, interactive=True)

                use_default_voice_cb.change(
                    fn=on_toggle_voice,
                    inputs=[use_default_voice_cb],
                    outputs=[custom_voice_input],
                )

                with gr.Accordion("⚙️ Dual-GPU Concurrency Tuning", open=False):
                    workers_slider = gr.Slider(
                        label="Concurrent ThreadPool Workers",
                        minimum=2,
                        maximum=8,
                        value=5,
                        step=1,
                        info="5 workers optimally saturates 2x T4 GPUs without queuing bottlenecks.",
                    )

                submit_btn = gr.Button("🚀 Start Saturated Dual-GPU Dubbing", variant="primary", size="lg")

            # RIGHT COLUMN: Audio Output & Live Throughput Logs
            with gr.Column(scale=5):
                gr.Markdown("### 🎧 2. Final Dubbed Master Output")
                audio_player = gr.Audio(label="Final Merged Audio Player", type="filepath")
                download_file = gr.File(label="📥 Download Master WAV")

                gr.Markdown("### 📊 3. Real-Time Throughput & Progress Log")
                logs_box = gr.Textbox(
                    label="Throughput & Progress Status Log",
                    lines=12,
                    autoscroll=True,
                    placeholder="Real-time chunk progress, throughput (chunks/sec), and cache hits will appear here...",
                )

        submit_btn.click(
            fn=run_high_throughput_srt_pipeline,
            inputs=[
                kaggle_url_input,
                srt_file_input,
                language_dropdown,
                use_default_voice_cb,
                custom_voice_input,
                workers_slider,
            ],
            outputs=[
                audio_player,
                download_file,
                logs_box,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    try:
        app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
    except ValueError:
        app.launch(show_api=False)

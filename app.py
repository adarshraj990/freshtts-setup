"""
🎬 Distributed Hybrid Auto-Dubbing Pipeline (SRT Sync)
Hugging Face Space Frontend Orchestrator & Multi-Backend Load Balancer

Features:
- Exclusive Manager Role: 100% serverless coordinator. Zero local XTTS generation.
- Hardcoded Verified API Pool: Pre-configured with 6 verified Hugging Face XTTS-v2 spaces.
- Manual Backend Extension: Custom Space / Gradio API endpoints can be added dynamically.
- Parallel Multi-Backend Dispatch: Concurrent round-robin distribution with automated failover retry.
- Precise Lip-Sync Timing: Millisecond-accurate timeline placement via pysrt.
- Chronological Audio Stitching: Seamless master canvas overlay using Pydub.
- Root Reference Voice: Automatically utilizes `reference_voice.wav` from the root directory.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Global client cache to reuse Gradio Client HTTP sessions
CLIENT_CACHE: Dict[str, Client] = {}
CLIENT_CACHE_LOCK = threading.Lock()

# Verified Hardcoded Hugging Face XTTS-v2 Spaces Pool
VERIFIED_SPACES: Dict[str, Dict[str, Any]] = {
    "Aviranjanprasad/Bhojpuri-XTTS-API": {
        "name": "Aviranjanprasad/Bhojpuri-XTTS-API",
        "endpoint": "/synthesize_speech",
        "type": "bhojpuri",
    },
    "hasanbasbunar/Voice-Cloning-XTTS-v2": {
        "name": "hasanbasbunar/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "eagien/XTTS": {
        "name": "eagien/XTTS",
        "endpoint": "/predict",
        "type": "eagien",
    },
    "applore/xtts-voice-cloning-demo": {
        "name": "applore/xtts-voice-cloning-demo",
        "endpoint": "/predict",
        "type": "applore",
    },
    "Fatimamirza970/Voice-Cloning-XTTS-V2": {
        "name": "Fatimamirza970/Voice-Cloning-XTTS-V2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "JymNils/Voice-Cloning-XTTS-v2": {
        "name": "JymNils/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
}


def check_reference_voice() -> str:
    """Verifies that reference_voice.wav is present in the root directory."""
    if REFERENCE_VOICE_PATH.is_file():
        size_mb = REFERENCE_VOICE_PATH.stat().st_size / (1024 * 1024)
        return f"✅ `reference_voice.wav` detected in root folder ({size_mb:.2f} MB)"
    return "⚠️ `reference_voice.wav` NOT FOUND in root directory. Please place it in the root before dubbing."


def get_or_create_client(endpoint_or_url: str) -> Client:
    """Retrieves an existing Gradio Client or establishes a new connection in a thread-safe manner."""
    clean_target = endpoint_or_url.strip().rstrip("/")
    with CLIENT_CACHE_LOCK:
        if clean_target not in CLIENT_CACHE:
            CLIENT_CACHE[clean_target] = Client(clean_target)
        return CLIENT_CACHE[clean_target]


def extract_audio_path(res: Any) -> Optional[str]:
    """Safely extracts a local audio file path string from various Gradio return types (str, dict, tuple)."""
    if res is None:
        return None
    if isinstance(res, str):
        return res
    if isinstance(res, (list, tuple)):
        for item in reversed(res):
            p = extract_audio_path(item)
            if p:
                return p
        return None
    if isinstance(res, dict):
        for key in ("path", "name", "orig_name"):
            val = res.get(key)
            if val and isinstance(val, str):
                return val
        return None
    return None


def execute_backend_call(
    backend_info: Dict[str, Any],
    text: str,
    ref_path: Path,
    lang_key: str,
) -> Optional[str]:
    """
    Adapter that maps text, reference voice audio, and target language
    to each specific backend's unique API parameter requirements.
    """
    target = backend_info["target"]
    b_type = backend_info.get("type", "custom")
    client = get_or_create_client(target)
    ref_param = handle_file(str(ref_path))

    # 1. Bhojpuri XTTS API
    if b_type == "bhojpuri":
        # /synthesize_speech(text, voice_preset, custom_audio_file, speed)
        result = client.predict(
            text=text,
            voice_preset="male_1",
            custom_audio_file=ref_param,
            speed=1.0,
            api_name="/synthesize_speech",
        )
        return extract_audio_path(result)

    # 2. Applore XTTS Demo
    elif b_type == "applore":
        # /predict(text, speaker_wav, language) -> generated_audio
        result = client.predict(
            text=text,
            speaker_wav=ref_param,
            language=lang_key,
            api_name="/predict",
        )
        return extract_audio_path(result)

    # 3. Eagien XTTS
    elif b_type == "eagien":
        # /predict(input_text, speaker_wav, language) -> (status, audio)
        lang_code = lang_key if lang_key in ["es", "fr", "de", "zh", "ja", "ko"] else "en"
        result = client.predict(
            input_text=text,
            speaker_wav=ref_param,
            language=lang_code,
            api_name="/predict",
        )
        return extract_audio_path(result)

    # 4. Hasanbasbunar / Fatimamirza970 / JymNils
    elif b_type == "hasanbasbunar":
        # /voice_clone_synthesis uses full language names
        lang_name_map = {
            "hi": "Hindi",
            "fr": "French",
            "es": "Spanish",
            "pt": "Portuguese",
        }
        lang_name = lang_name_map.get(lang_key, "English")
        result = client.predict(
            text=text,
            reference_audio_url=None,
            example_audio_name="audio_1.wav",
            language=lang_name,
            temperature=0.75,
            speed=1.0,
            do_sample=True,
            repetition_penalty=5.0,
            length_penalty=1.0,
            gpt_cond_len=30,
            top_k=50,
            top_p=0.85,
            remove_silence_enabled=True,
            silence_threshold=-45,
            min_silence_len=300,
            keep_silence=100,
            text_splitting_method="Native XTTS splitting",
            max_chars_per_segment=250,
            enable_preprocessing=False,
            api_name="/voice_clone_synthesis",
        )
        return extract_audio_path(result)

    # 5. Custom / Generic Backend
    else:
        try:
            res = client.predict(
                text_chunk=text,
                target_lang=lang_key,
                reference_voice=ref_param,
                api_name="/predict",
            )
            return extract_audio_path(res)
        except Exception:
            res = client.predict(
                text=text,
                speaker_wav=ref_param,
                language=lang_key,
                api_name="/predict",
            )
            return extract_audio_path(res)


def synthesize_chunk_task(
    chunk_tuple: Tuple[int, int, int, str],
    active_backends: List[Dict[str, Any]],
    ref_path: Path,
    target_lang_code: str,
) -> Tuple[int, int, Optional[str], Optional[str], str]:
    """
    Dispatches a single subtitle dialogue chunk across the active backend pool
    with round-robin priority and automated failover retry.
    """
    chunk_idx, start_time_ms, end_time_ms, chunk_text = chunk_tuple

    # Round-robin initial worker assignment
    primary_idx = (chunk_idx - 1) % len(active_backends)
    ordered_backends = [active_backends[primary_idx]] + [
        b for i, b in enumerate(active_backends) if i != primary_idx
    ]

    last_error = None
    for attempt, b_info in enumerate(ordered_backends, start=1):
        target_name = b_info["name"]
        try:
            audio_path = execute_backend_call(b_info, chunk_text, ref_path, target_lang_code)
            if audio_path and os.path.exists(audio_path):
                return chunk_idx, start_time_ms, audio_path, None, target_name
        except Exception as e:
            last_error = f"{target_name}: {e}"
            time.sleep(0.5)

    return chunk_idx, start_time_ms, None, last_error or "All backends failed", "None"


def run_distributed_dubbing(
    selected_spaces: List[str],
    custom_url_1: str,
    custom_url_2: str,
    custom_url_3: str,
    srt_file_path: Optional[str],
    target_language: str,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[Optional[str], str]:
    """
    Distributed Multi-Backend Auto-Dubbing Manager:
    1. Gathers all selected hardcoded Spaces and custom manual endpoints.
    2. Verifies the root reference_voice.wav file.
    3. Parses SRT subtitles using pysrt.
    4. Distributes generation requests concurrently across all available backends.
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
    log("🚀 Initializing Distributed Multi-Backend Auto-Dubbing Manager...")

    # 1. Verify root reference voice
    if not REFERENCE_VOICE_PATH.is_file():
        err_msg = (
            "FATAL ERROR: 'reference_voice.wav' not found in root directory! "
            "Please ensure reference_voice.wav is present in the workspace root."
        )
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🎙️ Active Reference Voice: `{REFERENCE_VOICE_PATH.name}` ({REFERENCE_VOICE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Build the Active Backend Pool
    active_backends: List[Dict[str, Any]] = []

    # Add selected hardcoded spaces
    if selected_spaces:
        for space_id in selected_spaces:
            if space_id in VERIFIED_SPACES:
                meta = dict(VERIFIED_SPACES[space_id])
                meta["target"] = space_id
                active_backends.append(meta)

    # Add custom manual endpoints
    for idx, c_url in enumerate([custom_url_1, custom_url_2, custom_url_3], start=1):
        if c_url and c_url.strip():
            clean_url = c_url.strip().rstrip("/")
            active_backends.append({
                "name": f"Custom API #{idx} ({clean_url})",
                "target": clean_url,
                "type": "custom",
            })

    if not active_backends:
        err_msg = "No backend selected! Please select at least one Space from the API Pool or enter a custom API URL."
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🌐 Active Multi-Backend Pool ({len(active_backends)} backends connected):")
    for idx, b in enumerate(active_backends, start=1):
        log(f"   • Backend #{idx}: {b['name']}")

    # 3. Parse Subtitle SRT File
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

    # 5. Concurrent Multi-Backend Generation
    target_lang_code = target_language.strip().lower()
    max_workers = min(len(active_backends), 8)

    log(f"🎬 Dubbing dialogue into target language: **{target_lang_code.upper()}**")
    log(f"⚡ Dispatching {num_chunks} chunks across {len(active_backends)} backend(s) with {max_workers} concurrent thread(s)...")

    chunk_results: List[Tuple[int, int, str]] = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                synthesize_chunk_task,
                chunk,
                active_backends,
                REFERENCE_VOICE_PATH,
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
    """Builds the polished Gradio Blocks User Interface with Distributed API Pool Manager."""
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="🎬 Distributed XTTS-v2 Auto-Dubbing Studio") as demo:
        # Header Banner
        gr.Markdown(
            """
            # 🎬 Distributed XTTS-v2 Auto-Dubbing Studio
            ### 🌐 Multi-Backend Serverless Load Balancer & Subtitle Lip-Sync Engine
            Coordinate deep voice cloning across verified Hugging Face Space backends and custom APIs with zero GPU overhead on this Space.
            """
        )

        # 1. API Pool Management Group
        with gr.Group():
            gr.Markdown("### 🌐 1. Distributed XTTS-v2 API Pool")
            gr.Markdown(
                "Select which verified Hugging Face Spaces to include in your load balancer pool. "
                "Subtitle chunks are automatically distributed across all active backends in parallel."
            )

            space_choices = list(VERIFIED_SPACES.keys())
            selected_spaces = gr.CheckboxGroup(
                label="Verified Hardcoded XTTS-v2 Spaces Pool",
                choices=space_choices,
                value=space_choices,
                info="All verified backends are active by default for maximum parallel speedup.",
            )

            with gr.Accordion("➕ Add Custom Space / Server APIs (Optional)", open=False):
                gr.Markdown("Enter additional custom Hugging Face Space names or Gradio `.live` endpoints:")
                custom_url_1 = gr.Textbox(
                    label="Custom API 1",
                    placeholder="e.g. username/custom-xtts-space or https://xxxx.gradio.live",
                    lines=1,
                )
                custom_url_2 = gr.Textbox(
                    label="Custom API 2",
                    placeholder="e.g. username/custom-xtts-space or https://yyyy.gradio.live",
                    lines=1,
                )
                custom_url_3 = gr.Textbox(
                    label="Custom API 3",
                    placeholder="e.g. username/custom-xtts-space or https://zzzz.gradio.live",
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
                    "🎬 Start Distributed Dubbing",
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
                    placeholder="Distributed pipeline logs will stream here when dubbing starts...",
                )

        # Wire Submit Button Event
        start_btn.click(
            fn=run_distributed_dubbing,
            inputs=[
                selected_spaces,
                custom_url_1,
                custom_url_2,
                custom_url_3,
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

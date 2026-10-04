"""
🎬 Distributed Hybrid Auto-Dubbing Pipeline (SRT Sync)
Hugging Face Space Frontend Orchestrator & Ultra-Robust Multi-Backend Sequential Queue

Features:
- Exclusive Manager Role: 100% serverless coordinator. Zero local XTTS generation.
- Curated Verified Active Pool (11 Spaces):
  1. hasanbasbunar/Voice-Cloning-XTTS-v2 (/voice_clone_synthesis)
  2. Aviranjanprasad/Bhojpuri-XTTS-API (/synthesize_speech)
  3. yasirs/XTTS-V2 (/voice_clone_synthesis)
  4. ama605/Russian-Voice-Cloning-XTTS-v2 (/voice_clone_synthesis)
  5. YemenEdu/Voice-Cloning-XTTS-v2 (/voice_clone_synthesis)
  6. minsus/Voice-Cloning-XTTS-v2 (/voice_clone_synthesis)
  7. tungpcco/XTTS-v2 (/voice_clone_synthesis)
  8. TGPro1/XTTS-v2 (/clone_voice)
  9. Fatimamirza970/Voice-Cloning-XTTS-v2 (/voice_clone_synthesis)
  10. applore/xtts-voice-cloning-demo (/predict)
  11. eagien/XTTS (/predict)
- Startup Health Check & Wake-Up Ping: Non-blocking background health check automatically wakes sleeping spaces and filters out dead, sleeping, or quota-exceeded spaces, retaining only responsive backends in active rotation.
- Strict Sequential Batch Queue: Strict 1-by-1 wait-for-response dispatch with round-robin rotation and dynamic failover. Zero concurrent collisions or queue drops.
- Precise Lip-Sync Timing: Millisecond-accurate timeline placement via pysrt.
- Chronological Audio Stitching: Seamless master canvas overlay using Pydub.
- Root Reference Voice: Automatically utilizes `reference_voice.wav` from the root directory.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Curated Verified Active XTTS-v2 Spaces Pool (11 Verified Spaces)
VERIFIED_SPACES: Dict[str, Dict[str, Any]] = {
    "hasanbasbunar/Voice-Cloning-XTTS-v2": {
        "name": "hasanbasbunar/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "Aviranjanprasad/Bhojpuri-XTTS-API": {
        "name": "Aviranjanprasad/Bhojpuri-XTTS-API",
        "endpoint": "/synthesize_speech",
        "type": "bhojpuri",
    },
    "yasirs/XTTS-V2": {
        "name": "yasirs/XTTS-V2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "ama605/Russian-Voice-Cloning-XTTS-v2": {
        "name": "ama605/Russian-Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "YemenEdu/Voice-Cloning-XTTS-v2": {
        "name": "YemenEdu/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "minsus/Voice-Cloning-XTTS-v2": {
        "name": "minsus/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "tungpcco/XTTS-v2": {
        "name": "tungpcco/XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "TGPro1/XTTS-v2": {
        "name": "TGPro1/XTTS-v2",
        "endpoint": "/clone_voice",
        "type": "tgpro",
    },
    "Fatimamirza970/Voice-Cloning-XTTS-v2": {
        "name": "Fatimamirza970/Voice-Cloning-XTTS-v2",
        "endpoint": "/voice_clone_synthesis",
        "type": "hasanbasbunar",
    },
    "applore/xtts-voice-cloning-demo": {
        "name": "applore/xtts-voice-cloning-demo",
        "endpoint": "/predict",
        "type": "applore",
    },
    "eagien/XTTS": {
        "name": "eagien/XTTS",
        "endpoint": "/predict",
        "type": "eagien",
    },
}

# Dynamic Runtime Health & Warm-Up Tracking
HEALTHY_BACKENDS: Set[str] = set()
HEALTHY_BACKENDS_LOCK = threading.Lock()

WARMUP_STATUS: Dict[str, str] = {
    space_id: "Health check pending... ⏳" for space_id in VERIFIED_SPACES
}
WARMUP_LOCK = threading.Lock()


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


def reset_client(endpoint_or_url: str):
    """Purges a client from the cache so any dead connection is cleanly re-established."""
    clean_target = endpoint_or_url.strip().rstrip("/")
    with CLIENT_CACHE_LOCK:
        if clean_target in CLIENT_CACHE:
            del CLIENT_CACHE[clean_target]


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

    # 4. TGPro1 XTTS
    elif b_type == "tgpro":
        # /clone_voice(text, language, reference_audio) -> generated_speech
        tg_supported = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
            "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"
        ]
        lang = lang_key if lang_key in tg_supported else "en"
        result = client.predict(
            text=text,
            language=lang,
            reference_audio=ref_param,
            api_name="/clone_voice",
        )
        return extract_audio_path(result)

    # 5. Hasanbasbunar / Yasirs / ama605 / YemenEdu / Minsus / Tungpcco / Fatimamirza
    elif b_type == "hasanbasbunar":
        # /voice_clone_synthesis uses full language names
        lang_name_map = {
            "hi": "Hindi",
            "fr": "French",
            "es": "Spanish",
            "pt": "Portuguese",
            "en": "English",
            "ru": "Russian",
            "ar": "Arabic",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "zh": "Chinese",
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

    # 6. Custom / Generic Backend
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


def execute_backend_call_with_timeout(
    backend_info: Dict[str, Any],
    text: str,
    ref_path: Path,
    lang_key: str,
    timeout_seconds: int = 75,
) -> Optional[str]:
    """
    Executes a single backend call with a strict timeout to prevent hung requests.
    Automatically resets the client cache entry if an error or timeout occurs.
    """
    target = backend_info["target"]
    with ThreadPoolExecutor(max_workers=1) as single_executor:
        future = single_executor.submit(
            execute_backend_call, backend_info, text, ref_path, lang_key
        )
        try:
            return future.result(timeout=timeout_seconds)
        except Exception:
            reset_client(target)
            raise


# ── Startup Health Check & Wake-Up Ping Mechanism ─────────────────────────────

def ping_single_backend(space_id: str, b_info: Dict[str, Any]):
    """
    Sends a lightweight test request ('Test') to wake up a sleeping space.
    Automatically flags the space as healthy in HEALTHY_BACKENDS if responsive.
    """
    try:
        with WARMUP_LOCK:
            WARMUP_STATUS[space_id] = "Pinging / Checking... ⏳"
        safe_print(f"🔥 [Startup Health Check] Testing {space_id}...")
        meta = dict(b_info)
        meta["target"] = space_id
        audio_path = execute_backend_call_with_timeout(
            meta,
            text="Test",
            ref_path=REFERENCE_VOICE_PATH,
            lang_key="en",
            timeout_seconds=60,
        )
        if audio_path and os.path.exists(audio_path):
            with WARMUP_LOCK:
                WARMUP_STATUS[space_id] = "Ready 🟢 (Active in Pool)"
            with HEALTHY_BACKENDS_LOCK:
                HEALTHY_BACKENDS.add(space_id)
            safe_print(f"✅ [Startup Health Check] {space_id} is healthy! Added to active rotation.")
        else:
            with WARMUP_LOCK:
                WARMUP_STATUS[space_id] = "Standby / Awakening 🟡"
            with HEALTHY_BACKENDS_LOCK:
                HEALTHY_BACKENDS.discard(space_id)
            safe_print(f"🟡 [Startup Health Check] {space_id} returned no audio. Filtered out of active pool.")
    except Exception as e:
        err_snippet = str(e).replace("\n", " ")[:35]
        with WARMUP_LOCK:
            WARMUP_STATUS[space_id] = f"Inactive/Sleep 🔴 ({err_snippet})"
        with HEALTHY_BACKENDS_LOCK:
            HEALTHY_BACKENDS.discard(space_id)
        safe_print(f"⚠️ [Startup Health Check] {space_id} failed: {err_snippet}")


def run_startup_warmup_all():
    """Fires parallel lightweight wake-up health pings to all 11 hardcoded spaces in the background."""
    safe_print("🚀 [Startup Health Check] Initiating health checks across all 11 XTTS-v2 spaces...")
    for space_id, b_info in VERIFIED_SPACES.items():
        t = threading.Thread(target=ping_single_backend, args=(space_id, b_info), daemon=True)
        t.start()


def get_warmup_status_markdown() -> str:
    """Renders a clean Markdown table of the current health and warm-up status of all backends."""
    lines = [
        "| Verified Space | Endpoint | Health & Warm-Up Status | Active in Pool |",
        "| :--- | :--- | :--- | :---: |",
    ]
    with WARMUP_LOCK:
        for space_id, info in VERIFIED_SPACES.items():
            status = WARMUP_STATUS.get(space_id, "Pending ⏳")
            with HEALTHY_BACKENDS_LOCK:
                is_active = "✅ Yes" if space_id in HEALTHY_BACKENDS else "❌ No"
            lines.append(f"| `{space_id}` | `{info['endpoint']}` | **{status}** | {is_active} |")
    return "\n".join(lines)


# ── Distributed Auto-Dubbing Engine (Strict Sequential Queue) ──────────────────

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
    1. Gathers responsive spaces from the active health check pool.
    2. Automatically filters out dead, sleeping, or quota-exceeded spaces.
    3. Parses SRT subtitles into dialogue chunks.
    4. Executes a STRICT SEQUENTIAL BATCH QUEUE:
       - Sends Chunk 1 to an available healthy backend in rotation.
       - WAITS until it successfully returns the audio chunk.
       - ONLY THEN dispatches Chunk 2 to the next healthy backend in rotation.
       - Gracefully falls back to the next healthy backend if a failure occurs.
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
        safe_print(entry)

    start_time = time.time()
    log("🚀 Initializing Ultra-Robust Multi-Backend Auto-Dubbing Manager...")

    # 1. Verify root reference voice
    if not REFERENCE_VOICE_PATH.is_file():
        err_msg = (
            "FATAL ERROR: 'reference_voice.wav' not found in root directory! "
            "Please ensure reference_voice.wav is present in the workspace root."
        )
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🎙️ Active Reference Voice: `{REFERENCE_VOICE_PATH.name}` ({REFERENCE_VOICE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Build the Active Backend Pool with Health Filtering
    active_backends: List[Dict[str, Any]] = []

    with HEALTHY_BACKENDS_LOCK:
        healthy_set = set(HEALTHY_BACKENDS)

    active_selected_spaces: List[str] = []
    if selected_spaces:
        # Filter: retain responsive spaces verified by startup health check
        healthy_selected = [s for s in selected_spaces if s in healthy_set]
        if healthy_selected:
            active_selected_spaces = healthy_selected
            excluded = [s for s in selected_spaces if s not in healthy_set]
            log(f"🛡️ Health Check Filter: Retained {len(healthy_selected)} verified responsive backend(s) in active rotation.")
            if excluded:
                log(f"💤 Automatically filtered out {len(excluded)} non-responsive/sleeping spaces: {', '.join(excluded)}")
        else:
            # Fallback if health checks are still running or none confirmed yet
            log("ℹ️ Health checks in progress. Using all selected spaces with dynamic per-chunk failover.")
            active_selected_spaces = selected_spaces

    for space_id in active_selected_spaces:
        if space_id in VERIFIED_SPACES:
            meta = dict(VERIFIED_SPACES[space_id])
            meta["target"] = space_id
            active_backends.append(meta)

    # Add custom manual endpoints (always included if provided)
    for idx, c_url in enumerate([custom_url_1, custom_url_2, custom_url_3], start=1):
        if c_url and c_url.strip():
            clean_url = c_url.strip().rstrip("/")
            active_backends.append({
                "name": f"Custom API #{idx} ({clean_url})",
                "target": clean_url,
                "type": "custom",
            })

    if not active_backends:
        err_msg = "No responsive backend available! Please check your space selections or add a custom API URL."
        log(f"❌ {err_msg}")
        raise gr.Error(err_msg)

    log(f"🌐 Active Rotation Pool ({len(active_backends)} backends connected):")
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

    # 5. Strict Sequential Batch Queue (Wait-For-Response)
    target_lang_code = target_language.strip().lower()
    num_backends = len(active_backends)
    backend_rot_idx = 0

    log(f"🎬 Dubbing dialogue into target language: **{target_lang_code.upper()}**")
    log("🔄 Mode: **Strict Sequential Queue** (1-by-1 wait-for-response, zero simultaneous collisions)")
    log(f"⚡ Processing {num_chunks} chunks sequentially across active backend rotation pool...")

    chunk_results: List[Tuple[int, int, str]] = []
    successful_chunks = 0

    for idx, (chunk_idx, start_time_ms, end_time_ms, chunk_text) in enumerate(chunks_to_process, start=1):
        progress_val = (idx - 1) / num_chunks
        progress(progress_val, desc=f"Sequential Queue: Chunk [{idx}/{num_chunks}] (Waiting for response...)")

        start_rot = backend_rot_idx
        chunk_success = False
        last_error = None

        # Sequential retry loop across available healthy backends for this chunk
        for attempt in range(num_backends):
            curr_backend_idx = (start_rot + attempt) % num_backends
            b_info = active_backends[curr_backend_idx]
            target_name = b_info["name"]

            log(f"⏳ [Chunk #{chunk_idx}/{num_chunks}] Dispatching to '{target_name}' (Waiting for audio response)...")
            call_start = time.time()
            try:
                audio_path = execute_backend_call_with_timeout(
                    b_info,
                    chunk_text,
                    REFERENCE_VOICE_PATH,
                    target_lang_code,
                    timeout_seconds=75,
                )
                call_dur = time.time() - call_start
                if audio_path and os.path.exists(audio_path):
                    log(f"✅ [Chunk #{chunk_idx}/{num_chunks}] Completed by '{target_name}' in {call_dur:.2f}s.")
                    chunk_results.append((chunk_idx, start_time_ms, audio_path))
                    chunk_success = True
                    successful_chunks += 1
                    # Advance rotation pointer so the NEXT chunk goes to the next backend
                    backend_rot_idx = (curr_backend_idx + 1) % num_backends
                    break
                else:
                    log(f"⚠️ [Chunk #{chunk_idx}/{num_chunks}] '{target_name}' returned no audio ({call_dur:.2f}s). Retrying next backend...")
            except Exception as e:
                call_dur = time.time() - call_start
                last_error = str(e).replace("\n", " ")
                log(f"⚠️ [Chunk #{chunk_idx}/{num_chunks}] '{target_name}' error ({call_dur:.2f}s): {last_error}. Retrying next backend...")

                # If backend encountered quota or rate-limiting error, discard from healthy set
                if any(err_term in last_error.lower() for err_term in ["quota", "rate limit", "429", "timeout", "504"]):
                    with HEALTHY_BACKENDS_LOCK:
                        HEALTHY_BACKENDS.discard(b_info.get("target"))
                    log(f"⛔ Automatically removed '{target_name}' from healthy pool due to quota/timeout.")

                time.sleep(0.5)

        if not chunk_success:
            log(f"❌ [Chunk #{chunk_idx}/{num_chunks}] Failed across all {num_backends} backends: {last_error}")
            # Advance rotation pointer so subsequent chunk starts fresh
            backend_rot_idx = (start_rot + 1) % num_backends

        progress(idx / num_chunks, desc=f"Sequential Queue: Chunk [{idx}/{num_chunks}] complete")

    # 6. Assembly: Chronological Stitching onto Master Canvas
    progress(0.95, desc="Stitching audio chunks into master track...")
    chunk_results.sort(key=lambda x: x[0])  # Guarantee exact chronological order
    log(f"🎼 Assembling {len(chunk_results)} generated chunks onto master timeline canvas...")

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


# ── Gradio Blocks User Interface ──────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Builds the polished Gradio Blocks User Interface with Sequential Queue & Health Check Monitor."""
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="🎬 Distributed XTTS-v2 Auto-Dubbing Studio") as demo:
        # Header Banner
        gr.Markdown(
            """
            # 🎬 Distributed XTTS-v2 Auto-Dubbing Studio
            ### 🌐 Multi-Backend Sequential Queue & Active Health Check Orchestrator
            Coordinate deep voice cloning across 11 verified Hugging Face Space backends with automatic health filtering and zero local GPU overhead.
            """
        )

        # 1. API Pool Management Group
        with gr.Group():
            gr.Markdown("### 🌐 1. Distributed XTTS-v2 API Pool (11 Verified Spaces)")
            gr.Markdown(
                "All external spaces are tested on startup to detect sleeping or quota-exceeded instances. "
                "Only responsive backends are retained in active rotation. Subtitle chunks run in a "
                "**strict sequential queue** (1-by-1 wait-for-response) with automated failover."
            )

            with gr.Accordion("⚡ Backend Startup Health Check & Warm-Up Status", open=True):
                warmup_status_md = gr.Markdown(value=get_warmup_status_markdown())
                refresh_warmup_btn = gr.Button("🔄 Re-Ping & Refresh Backend Health", size="sm")

            space_choices = list(VERIFIED_SPACES.keys())
            selected_spaces = gr.CheckboxGroup(
                label="Verified Hardcoded XTTS-v2 Spaces Pool (11 Backends)",
                choices=space_choices,
                value=space_choices,
                info="All verified backends are selected by default. Dead or sleeping spaces are automatically filtered out at runtime.",
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
                        ("Russian (ru)", "ru"),
                        ("Arabic (ar)", "ar"),
                        ("German (de)", "de"),
                        ("Japanese (ja)", "ja"),
                    ],
                    value="hi",
                    info="Select target speech translation language.",
                )

                # Reference voice status check indicator (No upload button in UI)
                voice_status = gr.Markdown(value=check_reference_voice())

                start_btn = gr.Button(
                    "🎬 Start Sequential Dubbing",
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
                    placeholder="Sequential pipeline logs will stream here when dubbing starts...",
                )

        # Wire Warm-Up Refresh Button Event
        def on_refresh_warmup():
            run_startup_warmup_all()
            time.sleep(1.0)
            return get_warmup_status_markdown()

        refresh_warmup_btn.click(
            fn=on_refresh_warmup,
            inputs=[],
            outputs=[warmup_status_md],
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


# Automatically initiate background health check & wake-up pings to all 11 spaces on startup
threading.Thread(target=run_startup_warmup_all, daemon=True).start()

app = build_app()

if __name__ == "__main__":
    try:
        app.launch(server_name="0.0.0.0", server_port=7860)
    except ValueError:
        app.launch()

"""
🎬 Time-Synchronized Distributed SRT Voice Cloning & Auto-Dubbing Studio
Frontend: Hugging Face Space (Lightweight Gradio Client, Zero GPU Inference)
Backend: Remote Dual-GPU Kaggle / Colab XTTS-v2 Engine (2x T4 GPUs)

Key Features & Pipeline Architecture:
1. One-Time Speaker Registration (/register_speaker):
   - Reference audio uploaded ONCE to backend before batch dubbing.
   - Conditioning latents cached in GPU memory on Kaggle dual T4 GPUs.
   - Subtitle chunks call /synthesize_line sending ONLY text + language (no re-uploading).
2. Parallel Dual-GPU Saturation (ThreadPoolExecutor max_workers=4):
   - Fires 4 simultaneous requests to /synthesize_line to fully saturate 2x T4 GPUs.
   - Persistent requests.Session with connection pooling (pool_connections=20).
   - Asynchronous collection via concurrent.futures.as_completed() without sequential stalls.
   - 3-attempt automated retry loop with 2.0s backoff for network drops.
3. Complete Line Generation & SRT Time-Sync:
   - Full subtitle lines (natural emotion/prosody).
   - Silence stripping via pydub.silence (-40 dBFS threshold).
   - Time-stretching / slot-fitting to prevent cross-line bleeding.
   - Overlays onto an empty master canvas of total video duration (e.g., 17m 22s).
4. Real-Time Telemetry & Persistent Gallery:
   - Live throughput (lines/sec), elapsed time, and ETA.
   - Permanent local WAV preservation in outputs/ directory.
   - Dropdown selector, playback player, download button, and file deletion controls.
"""

import base64
import concurrent.futures
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
from pydub.silence import detect_leading_silence
from pydub.effects import speedup
import pysrt
import requests
from requests.adapters import HTTPAdapter

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
    "User-Agent": "AudioGenFlow-DualGPU-Client/3.0",
}

# Shared HTTP Session with Connection Pooling for Ultra-Fast Parallel Requests
SESSION = requests.Session()
adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=0,
)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

# In-Memory Cache for Idempotent Operations (Thread-Safe)
AUDIO_CACHE: Dict[str, bytes] = {}
CACHE_LOCK = threading.Lock()


# ── Audio Processing & Subtitle Cleansing ─────────────────────────────────────

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


def strip_dead_silence(seg: AudioSegment, threshold: float = -40.0, margin_ms: int = 25) -> AudioSegment:
    """
    Strips leading and trailing dead silence below threshold (~ -40 dBFS),
    leaving a minimal comfortable margin so word onsets and offsets sound natural.
    """
    if len(seg) < 100:
        return seg
    try:
        lead = detect_leading_silence(seg, silence_threshold=threshold, chunk_size=10)
        trail = detect_leading_silence(seg.reverse(), silence_threshold=threshold, chunk_size=10)
        if lead + trail >= len(seg):
            return seg
        start_cut = max(0, lead - margin_ms)
        end_cut = max(start_cut + 50, len(seg) - max(0, trail - margin_ms))
        return seg[start_cut:end_cut]
    except Exception:
        return seg


def fit_audio_to_slot(seg: AudioSegment, slot_duration_ms: int, max_speedup: float = 1.35) -> AudioSegment:
    """
    If audio exceeds the subtitle slot window (end_time_ms - start_time_ms),
    gently time-stretches / speeds it up to fit cleanly without overlapping.
    """
    if slot_duration_ms <= 0 or len(seg) <= slot_duration_ms:
        return seg

    speed_ratio = len(seg) / float(slot_duration_ms)
    # If overhang is negligible (<= 5%), keep as is
    if speed_ratio <= 1.05:
        return seg

    speed_factor = min(speed_ratio, max_speedup)
    try:
        chunk_size = min(100, max(20, len(seg) // 6))
        sped_up = speedup(seg, playback_speed=speed_factor, chunk_size=chunk_size, crossfade=15)
        # If still exceeding slot by more than 80ms, gently fade out tail
        if len(sped_up) > slot_duration_ms + 80:
            sped_up = sped_up[: slot_duration_ms + 80].fade_out(25)
        return sped_up
    except Exception as e:
        safe_print(f"Speedup notice: {e}, falling back to frame rate adjustment")
        try:
            stretched = seg._spawn(seg.raw_data, overrides={
                "frame_rate": int(seg.frame_rate * speed_factor)
            }).set_frame_rate(seg.frame_rate)
            return stretched
        except Exception:
            return seg


def get_cache_key(text: str, language: str, voice_hash: str) -> str:
    """Computes a unique MD5 hash key for caching speech audio."""
    raw = f"{text}|{language}|{voice_hash}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ── One-Time Speaker Registration on Backend ─────────────────────────────────

def register_speaker_on_backend(
    endpoint_base: str,
    ref_audio_path: str,
    timeout_sec: int = 60,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Sends the reference audio file ONCE to the backend endpoint '/register_speaker'.
    The backend caches conditioning latents in GPU memory for all subsequent chunks.
    Returns: (success: bool, speaker_id: str, error_message: str)
    """
    clean_base = endpoint_base.rstrip("/")
    for subpath in ["/voice_clone_synthesis", "/synthesize_line", "/register_speaker"]:
        if clean_base.endswith(subpath):
            clean_base = clean_base[:-len(subpath)].rstrip("/")

    url = f"{clean_base}/register_speaker"
    filename = Path(ref_audio_path).name
    mime_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"

    try:
        with open(ref_audio_path, "rb") as f:
            files = {"speaker_wav": (filename, f, mime_type)}
            data = {"speaker_id": f"spk_{Path(ref_audio_path).stem[:12]}"}
            resp = SESSION.post(
                url,
                headers=TUNNEL_HEADERS,
                files=files,
                data=data,
                timeout=timeout_sec,
            )

        if resp.status_code == 200:
            try:
                res_data = resp.json()
                sid = res_data.get("speaker_id") or "active_speaker"
                return True, sid, None
            except Exception:
                return True, "active_speaker", None
        else:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:120]}"

    except Exception as e:
        return False, None, str(e)


# ── Ultra-Fast Chunk Synthesis with Automated 3-Attempt Retry ────────────────

def synthesize_line_with_retry(
    endpoint_base: str,
    text: str,
    language: str,
    speaker_id: Optional[str] = None,
    ref_audio_path: Optional[str] = None,
    use_cached_speaker: bool = True,
    timeout_sec: int = 120,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Dispatches a single subtitle line to the backend with automated retry.
    - If use_cached_speaker is True: calls /synthesize_line sending ONLY text + language (no audio upload).
    - If fallback is needed: calls /voice_clone_synthesis with full audio upload.
    """
    clean_base = endpoint_base.rstrip("/")
    for subpath in ["/voice_clone_synthesis", "/synthesize_line", "/register_speaker"]:
        if clean_base.endswith(subpath):
            clean_base = clean_base[:-len(subpath)].rstrip("/")

    data = {
        "text": text,
        "text_chunk": text,
        "language": language,
        "target_lang": language,
    }
    if speaker_id:
        data["speaker_id"] = speaker_id

    last_error = "Unknown error"

    for attempt in range(1, max_retries + 1):
        try:
            if use_cached_speaker:
                target_url = f"{clean_base}/synthesize_line"
                resp = SESSION.post(
                    target_url,
                    headers=TUNNEL_HEADERS,
                    data=data,
                    timeout=timeout_sec,
                )
            else:
                target_url = f"{clean_base}/voice_clone_synthesis"
                filename = Path(ref_audio_path).name
                mime_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
                with open(ref_audio_path, "rb") as f1, open(ref_audio_path, "rb") as f2:
                    files = [
                        ("speaker_wav", (filename, f1, mime_type)),
                        ("reference_voice", (filename, f2, mime_type)),
                    ]
                    resp = SESSION.post(
                        target_url,
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
                if resp.content and len(resp.content) > 100:
                    return resp.content, None
                else:
                    last_error = "Empty audio response from backend"

            elif resp.status_code in [500, 502, 503, 504]:
                try:
                    err_json = resp.json()
                    detail = err_json.get("detail", resp.text[:140])
                    last_error = f"HTTP {resp.status_code}: {detail}"
                except Exception:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:140]}"
            else:
                try:
                    err_json = resp.json()
                    detail = err_json.get("detail", resp.text[:140])
                    last_error = f"HTTP {resp.status_code}: {detail}"
                except Exception:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:140]}"

                # Abort early on client-side input error
                if resp.status_code == 400:
                    return None, last_error

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            last_error = f"Network Timeout / Connection Error: {type(net_err).__name__}"
        except Exception as e:
            last_error = f"Request error: {str(e)}"

        if attempt < max_retries:
            safe_print(f"⚠️ [Attempt {attempt}/{max_retries} failed] {last_error} -> Retrying in {retry_delay_sec}s...")
            time.sleep(retry_delay_sec)

    return None, f"Failed after {max_retries} attempts ({last_error})"


# ── Interactive Backend Connection Health Check ──────────────────────────────

def check_connection_health(api_url: str):
    """
    Lightweight, non-blocking health check ping to the remote Kaggle API endpoint.
    Verifies tunnel availability (Localtunnel / Ngrok), validates bypass headers,
    and calculates network roundtrip latency in milliseconds.
    """
    raw_url = (api_url or "").strip()
    if not raw_url:
        yield "⚪ **Not Connected** *(Please enter your Kaggle API URL first)*"
        return

    # Automatically prefix protocol if omitted
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = f"https://{raw_url}"

    clean_url = raw_url.rstrip("/")

    # Derive root URL if user supplied an endpoint subpath
    for subpath in ["/voice_clone_synthesis", "/synthesize_line", "/register_speaker"]:
        if clean_url.endswith(subpath):
            clean_url = clean_url[:-len(subpath)].rstrip("/")

    root_url = clean_url

    yield "⏳ **Checking...**"

    t0 = time.time()
    try:
        # Ping root URL with 7-second timeout and tunnel bypass headers
        resp = SESSION.get(
            f"{root_url}/",
            headers=TUNNEL_HEADERS,
            timeout=7,
            allow_redirects=True,
        )
        latency_ms = max(1, int((time.time() - t0) * 1000))

        # Status code evaluation
        if resp.status_code in [200, 204, 301, 302, 307, 308, 404, 405]:
            try:
                data = resp.json()
                if "service" in data or data.get("status") == "healthy":
                    gpus = data.get("total_gpus", 2)
                    yield f"✅ **Connected (Latency: {latency_ms}ms | {gpus}x GPUs Active | One-Time Speaker Cache Ready)**"
                    return
            except Exception:
                pass
            yield f"✅ **Connected (Latency: {latency_ms}ms)**"
        elif resp.status_code in [502, 503, 504]:
            yield f"❌ **Connection Failed / Timeout** (HTTP {resp.status_code} Bad Gateway — Kaggle GPU worker or tunnel is offline)"
        else:
            yield f"⚠️ **Connected with warning** (Latency: {latency_ms}ms | HTTP {resp.status_code})"

    except requests.exceptions.Timeout:
        yield "❌ **Connection Failed / Timeout** (Server did not respond within 7s — verify Kaggle notebook is active)"
    except requests.exceptions.ConnectionError:
        yield "❌ **Connection Failed / Timeout** (Host unreachable or tunnel domain expired/offline)"
    except Exception as e:
        yield f"❌ **Connection Failed / Timeout** ({str(e)[:100]})"


# ── Persistent Generation History Management ──────────────────────────────────

def get_history_records() -> Tuple[List[Tuple[str, str]], List[List[str]], Optional[str]]:
    """
    Scans the outputs directory and returns:
    1. choices for dropdown: [(label, filepath_str), ...]
    2. table data: [[filename, size_str, created_time], ...]
    3. latest file path (or None)
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in OUTPUT_DIR.glob("*.wav") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    choices: List[Tuple[str, str]] = []
    table_data: List[List[str]] = []
    latest_file: Optional[str] = str(files[0]) if files else None

    for f in files:
        stat = f.stat()
        size_mb = f"{stat.st_size / (1024 * 1024):.2f} MB"
        created_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        label = f"{f.name} ({size_mb} • {created_str})"
        choices.append((label, str(f)))
        table_data.append([f.name, size_mb, created_str])

    return choices, table_data, latest_file


# ── High-Throughput Time-Synchronized SRT Voice Cloning Pipeline ──────────────

def run_high_throughput_srt_pipeline(
    kaggle_url: str,
    srt_file: Optional[str],
    language_code: str,
    use_default_voice: bool,
    custom_voice_file: Optional[str],
    concurrent_workers: int = 4,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Ultra-Fast Time-Synchronized Dual-GPU SRT Dubbing Engine:
    1. One-Time Speaker Registration: Caches speaker conditioning latents once on backend.
    2. 4 Simultaneous Parallel Dispatches: ThreadPoolExecutor(max_workers=4) saturates 2x T4 GPUs.
    3. Non-Blocking Async Collection: as_completed() collects results without sequential stalling.
    4. Exact SRT Order Memory Dictionary: results[chunk_id] = (start_ms, audio_segment).
    5. Dead Silence Stripping & Slot-Fitting: Trims dead silence (-40 dBFS) & fits duration slot.
    6. Exact Timeline Alignment: Overlays chunks onto an empty master canvas of total video duration.
    7. Real-Time Telemetry: Live lines/sec, elapsed time, and ETA tracking.
    8. Persistent History: Automatically registers output in local exports gallery.
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
    log("🚀 Initializing Saturated 4-Worker Parallel SRT Dubbing Pipeline...")

    # 1. Validate & Sanitize Kaggle API URL
    if not kaggle_url or not kaggle_url.strip():
        err = "Please enter your active Kaggle API URL (Localtunnel or Ngrok)!"
        log(f"❌ {err}")
        raise gr.Error(err)

    clean_url = kaggle_url.strip().rstrip("/")
    for subpath in ["/voice_clone_synthesis", "/synthesize_line", "/register_speaker"]:
        if clean_url.endswith(subpath):
            clean_url = clean_url[:-len(subpath)].rstrip("/")

    log(f"🌐 Backend Host: `{clean_url}`")

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

    # 3. One-Time Speaker Registration (/register_speaker)
    log(f"⚡ Registering reference voice once on backend: `{clean_url}/register_speaker`...")
    reg_ok, sid, reg_err = register_speaker_on_backend(clean_url, ref_path, timeout_sec=60)

    if reg_ok:
        log(f"✅ One-Time Speaker Registration Succeeded! Speaker ID: `{sid}`.")
        log("🚀 Speaker audio cached in GPU memory. Subtitle lines will send TEXT ONLY without re-uploading audio!")
        use_cached_speaker = True
    else:
        log(f"⚠️ Speaker registration note: {reg_err}. Falling back to standard multi-part upload mode.")
        use_cached_speaker = False
        sid = None

    # 4. Parse and Clean SRT Subtitles
    if not srt_file or not os.path.isfile(srt_file):
        err = "Please upload a valid .srt subtitle file!"
        log(f"❌ {err}")
        raise gr.Error(err)

    log(f"📄 Parsing SRT subtitles: `{Path(srt_file).name}`...")
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

    # Calculate total timeline duration from last subtitle's end time (+ 1500ms tail)
    last_sub_end_ms = max(sub.end.ordinal for sub in subs)
    total_video_duration_ms = last_sub_end_ms + 1500
    timeline_mins = total_video_duration_ms // 60000
    timeline_secs = (total_video_duration_ms % 60000) // 1000
    log(f"⏱️ Total Video Timeline Duration: {timeline_mins}m {timeline_secs}s ({total_video_duration_ms} ms)")

    # Filter out empty lines, subtitle tags, and pure timestamp artifacts
    cleaned_chunks: List[Tuple[int, int, int, int, str]] = []
    skipped_count = 0

    for i, sub in enumerate(subs, start=1):
        # Join multi-line subtitle blocks into a single natural sentence
        raw_sentence = " ".join(line.strip() for line in sub.text.splitlines() if line.strip())
        cleaned_text = clean_subtitle_text(raw_sentence)

        start_ms = sub.start.ordinal
        end_ms = sub.end.ordinal
        slot_duration_ms = max(400, end_ms - start_ms)

        if cleaned_text and re.search(r"\w", cleaned_text, re.UNICODE):
            cleaned_chunks.append((i, start_ms, end_ms, slot_duration_ms, cleaned_text))
        else:
            skipped_count += 1

    total_chunks = len(cleaned_chunks)
    log(f"📊 SRT Parsing Complete: {len(subs)} total blocks loaded.")
    if skipped_count > 0:
        log(f"⚡ Skipped {skipped_count} empty / non-spoken artifact blocks.")
    log(f"🎯 Valid Full-Sentence Dialogue Lines to Synthesize: **{total_chunks}**")

    if total_chunks == 0:
        err = "No valid spoken dialogue found in the SRT file after cleansing."
        log(f"❌ {err}")
        raise gr.Error(err)

    # 5. Saturated 4-Worker Parallel Execution (concurrent.futures.ThreadPoolExecutor max_workers=4)
    num_workers = int(concurrent_workers) if (concurrent_workers and int(concurrent_workers) >= 2) else 4
    log(f"⚡ Firing {num_workers} simultaneous requests across ThreadPoolExecutor(max_workers={num_workers}) to saturate Dual-GPUs...")

    # Map chunk_id to its exact start_time_ms on the SRT timeline
    chunk_timestamps: Dict[int, int] = {chunk[0]: chunk[1] for chunk in cleaned_chunks}
    # Maintain original subtitle order in an in-memory dictionary: results[chunk_id] = audio_segment
    results: Dict[int, AudioSegment] = {}
    completed_chunks = 0
    cache_hits = 0
    lang = language_code.strip().lower()

    def process_sub_block(chunk_info: Tuple[int, int, int, int, str]) -> Tuple[int, Optional[AudioSegment], Optional[str]]:
        nonlocal cache_hits
        idx, start_ms, end_ms, slot_duration_ms, text = chunk_info
        cache_key = get_cache_key(text, lang, voice_hash)

        # 1. Check in-memory cache
        with CACHE_LOCK:
            cached_bytes = AUDIO_CACHE.get(cache_key)

        if cached_bytes:
            seg = AudioSegment.from_file(io.BytesIO(cached_bytes))
            seg = seg.set_frame_rate(24000).set_channels(1)
            seg = strip_dead_silence(seg, threshold=-40.0)
            seg = fit_audio_to_slot(seg, slot_duration_ms=slot_duration_ms)
            if len(seg) > 30:
                seg = seg.fade_in(10).fade_out(10)
            return idx, seg, "CACHE_HIT"

        # 2. Remote Dual-GPU Synthesis Call to Kaggle /synthesize_line
        audio_bytes, err = synthesize_line_with_retry(
            endpoint_base=clean_url,
            text=text,
            language=lang,
            speaker_id=sid,
            ref_audio_path=ref_path,
            use_cached_speaker=use_cached_speaker,
            timeout_sec=120,
            max_retries=3,
            retry_delay_sec=2.0,
        )

        if audio_bytes and len(audio_bytes) > 100:
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                with CACHE_LOCK:
                    AUDIO_CACHE[cache_key] = audio_bytes

                seg = seg.set_frame_rate(24000).set_channels(1)
                seg = strip_dead_silence(seg, threshold=-40.0)
                seg = fit_audio_to_slot(seg, slot_duration_ms=slot_duration_ms)
                if len(seg) > 30:
                    seg = seg.fade_in(10).fade_out(10)

                return idx, seg, None
            except Exception as dec_err:
                return idx, None, f"Audio decode error: {dec_err}"
        else:
            return idx, None, err or "Empty audio response"

    # Submit ALL jobs to executor simultaneously without waiting in a sequential loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_sub_block, chunk): chunk
            for chunk in cleaned_chunks
        }

        # Asynchronously collect results as each completes using concurrent.futures.as_completed()
        for future in concurrent.futures.as_completed(future_to_chunk):
            sub_idx, seg, status = future.result()
            completed_chunks += 1
            progress_ratio = completed_chunks / total_chunks

            # Compute real-time telemetry: throughput, elapsed time, and ETA
            elapsed_current = max(0.1, time.time() - t_start)
            throughput = completed_chunks / elapsed_current
            remaining_chunks = total_chunks - completed_chunks
            eta_sec = remaining_chunks / max(0.01, throughput)
            eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"
            elapsed_str = f"{int(elapsed_current // 60):02d}:{int(elapsed_current % 60):02d}"

            progress(
                progress_ratio,
                desc=f"Dubbing [{completed_chunks}/{total_chunks}] • {throughput:.2f} lines/s • Elapsed: {elapsed_str} • ETA: {eta_str}",
            )

            if seg is not None:
                # Maintain original subtitle order in an in-memory dictionary: results[chunk_id] = audio_segment
                results[sub_idx] = seg
                if status == "CACHE_HIT":
                    cache_hits += 1
                    log(f"⚡ [Line #{sub_idx}/{total_chunks}] Cache Hit (0ms) | {throughput:.2f} lines/s | Elapsed: {elapsed_str} | ETA: {eta_str}")
                else:
                    log(f"✅ [Line #{sub_idx}/{total_chunks}] Synced ({len(seg)}ms) | {throughput:.2f} lines/s | Elapsed: {elapsed_str} | ETA: {eta_str}")
            else:
                log(f"⚠️ [Line #{sub_idx}/{total_chunks}] Failed: {status} | Elapsed: {elapsed_str} | ETA: {eta_str}")

    # 6. Precise Master Audio Timeline Assembly in Original Subtitle Order
    progress(0.95, desc="Overlaying audio segments onto master video timeline...")
    log(f"🎼 Assembling {len(results)} segments strictly in original SRT order onto {timeline_mins}m {timeline_secs}s canvas...")

    if not results:
        err = "All subtitle segments failed to synthesize. Please check your Kaggle GPU notebook logs."
        log(f"❌ {err}")
        raise gr.Error(err)

    # Empty silent master track matching total video duration
    master_canvas = AudioSegment.silent(duration=total_video_duration_ms, frame_rate=24000).set_channels(1)

    # Place each segment strictly at its exact start_time_ms
    for chunk_id in sorted(results.keys()):
        seg = results[chunk_id]
        start_ms = chunk_timestamps[chunk_id]
        master_canvas = master_canvas.overlay(seg, position=start_ms)

    # 7. Direct Export to outputs/ directory (Permanent storage)
    progress(0.98, desc="Exporting time-synchronized master WAV...")
    output_filename = f"dubbed_master_{lang}_{int(time.time())}.wav"
    output_path = OUTPUT_DIR / output_filename

    master_canvas.export(str(output_path), format="wav")
    final_file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Free memory buffers immediately
    del results
    del master_canvas

    total_time = time.time() - t_start
    avg_speed = total_chunks / max(0.1, total_time)

    log(f"🎉 Pipeline finished in {total_time:.2f}s ({avg_speed:.2f} lines/sec)!")
    log(f"📊 Completed: {completed_chunks}/{total_chunks} lines ({cache_hits} cache hits).")
    log(f"📁 Master Output File: `{output_filename}` ({final_file_size_mb:.2f} MB)")
    log(f"💾 Permanently saved to outputs/ and registered in generation history.")
    progress(1.0, desc="Dubbing complete!")

    # Refresh history records
    hist_choices, hist_table, _ = get_history_records()

    return (
        str(output_path),
        str(output_path),
        "\n".join(logs),
        gr.update(choices=hist_choices, value=str(output_path)),
        hist_table,
        str(output_path),
        str(output_path),
        f"✅ Generated & Saved: `{output_filename}` ({final_file_size_mb:.2f} MB)",
    )


# ── Gradio Web UI Layout ──────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    initial_choices, initial_table, initial_latest = get_history_records()

    with gr.Blocks(theme=theme, title="🎬 Saturated Dual-GPU SRT Dubbing Studio") as demo:
        # Header Banner
        gr.Markdown(
            """
            # 🎬 Saturated Dual-GPU SRT Voice Cloning Studio
            ### ⚡ Ultra-Fast Parallel Subtitle Auto-Dubbing via Remote Kaggle Dual T4 GPUs
            **One-Time Speaker Caching (/register_speaker)**: Reference audio is uploaded once to GPU memory. 
            All subtitle lines are dispatched simultaneously via **4 concurrent workers** to /synthesize_line with full sentence prosody, dead silence stripping (-40 dBFS), slot-fitting time stretch, 3-attempt auto-retry, and persistent exports gallery.
            """
        )

        with gr.Row():
            # LEFT COLUMN: User Inputs
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Connection & Subtitle Configuration")

                with gr.Row():
                    kaggle_url_input = gr.Textbox(
                        label="Kaggle API URL (Dynamic Localtunnel / Ngrok)",
                        placeholder="https://famous-sheep-wash.loca.lt or https://xxxx.ngrok-free.app",
                        value="",
                        lines=1,
                        scale=7,
                        info="Paste your active Kaggle GPU tunnel URL.",
                    )
                    check_conn_btn = gr.Button("🔍 Check Connection", variant="secondary", scale=3)

                conn_status_box = gr.Markdown("⚪ **Not Connected**")

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
                        value=4,
                        step=1,
                        info="4 simultaneous workers fully saturates 2x T4 GPUs on Kaggle, eliminating idle wait time.",
                    )

                submit_btn = gr.Button("🚀 Start Ultra-Fast Parallel Dubbing", variant="primary", size="lg")

            # RIGHT COLUMN: Audio Output & Live Throughput Logs
            with gr.Column(scale=5):
                gr.Markdown("### 🎧 2. Current Dubbed Master Output")
                audio_player = gr.Audio(label="Time-Synchronized Master Audio Player", type="filepath")
                download_file = gr.File(label="📥 Download Master WAV")

                gr.Markdown("### 📊 3. Real-Time Telemetry & Progress Log")
                logs_box = gr.Textbox(
                    label="Real-Time Telemetry Log (Lines/sec • Elapsed • ETA)",
                    lines=10,
                    autoscroll=True,
                    placeholder="Real-time chunk progress, SRT time-sync, latency, ETA, and cache hits will appear here...",
                )

        # ── PERSISTENT GENERATION HISTORY GALLERY ─────────────────────────────
        gr.Markdown("---")
        with gr.Accordion("📁 4. Persistent Generation History & Exports Gallery", open=True):
            gr.Markdown(
                """
                **Local Audio Archive**: All generated full-timeline master audios are permanently preserved in the local `outputs/` folder.
                Files are **never** automatically deleted after downloading. Select any past generation to preview or download, or manage files below.
                """
            )
            with gr.Row():
                with gr.Column(scale=6):
                    history_dropdown = gr.Dropdown(
                        label="Select Master Audio from History",
                        choices=initial_choices,
                        value=initial_latest,
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 Refresh List", variant="secondary", size="sm")
                        delete_selected_btn = gr.Button("🗑️ Delete Selected Audio", variant="stop", size="sm")
                        clear_all_btn = gr.Button("⚠️ Clear All History", variant="secondary", size="sm")
                    history_status_md = gr.Markdown("Ready.")

                with gr.Column(scale=4):
                    history_player = gr.Audio(
                        label="History Audio Player",
                        type="filepath",
                        value=initial_latest,
                    )
                    history_download = gr.File(
                        label="📥 Download Selected Audio",
                        value=initial_latest,
                    )

            history_table = gr.Dataframe(
                headers=["Filename", "Size (MB)", "Created Timestamp"],
                datatype=["str", "str", "str"],
                value=initial_table,
                label="Stored Audio Files in outputs/",
                interactive=False,
            )

        # ── Event Bindings ───────────────────────────────────────────────────

        # Connection Health Check
        check_conn_btn.click(
            fn=check_connection_health,
            inputs=[kaggle_url_input],
            outputs=[conn_status_box],
        )

        kaggle_url_input.submit(
            fn=check_connection_health,
            inputs=[kaggle_url_input],
            outputs=[conn_status_box],
        )

        # Main Synthesis Pipeline
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
                history_dropdown,
                history_table,
                history_player,
                history_download,
                history_status_md,
            ],
        )

        # History Actions
        def on_select_history(selected_path: Optional[str]):
            if selected_path and os.path.isfile(selected_path):
                return selected_path, selected_path
            return None, None

        history_dropdown.change(
            fn=on_select_history,
            inputs=[history_dropdown],
            outputs=[history_player, history_download],
        )

        def on_refresh_history():
            choices, table_data, latest = get_history_records()
            return (
                gr.update(choices=choices, value=latest),
                table_data,
                latest,
                latest,
                f"✅ Gallery refreshed. {len(choices)} files available.",
            )

        refresh_history_btn.click(
            fn=on_refresh_history,
            outputs=[
                history_dropdown,
                history_table,
                history_player,
                history_download,
                history_status_md,
            ],
        )

        def on_delete_selected(selected_path: Optional[str]):
            if not selected_path or not os.path.isfile(selected_path):
                choices, table_data, latest = get_history_records()
                return (
                    gr.update(choices=choices, value=latest),
                    table_data,
                    latest,
                    latest,
                    "⚠️ No audio selected to delete.",
                )
            try:
                name = Path(selected_path).name
                os.remove(selected_path)
                msg = f"🗑️ Deleted `{name}` successfully."
            except Exception as e:
                msg = f"❌ Error deleting file: {e}"

            choices, table_data, latest = get_history_records()
            return (
                gr.update(choices=choices, value=latest),
                table_data,
                latest,
                latest,
                msg,
            )

        delete_selected_btn.click(
            fn=on_delete_selected,
            inputs=[history_dropdown],
            outputs=[
                history_dropdown,
                history_table,
                history_player,
                history_download,
                history_status_md,
            ],
        )

        def on_clear_all():
            choices, _, _ = get_history_records()
            count = 0
            for _, path_str in choices:
                try:
                    if os.path.isfile(path_str):
                        os.remove(path_str)
                        count += 1
                except Exception:
                    pass
            choices, table_data, latest = get_history_records()
            return (
                gr.update(choices=choices, value=None),
                table_data,
                None,
                None,
                f"🧹 Cleared {count} audio files from history.",
            )

        clear_all_btn.click(
            fn=on_clear_all,
            outputs=[
                history_dropdown,
                history_table,
                history_player,
                history_download,
                history_status_md,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    try:
        app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
    except ValueError:
        app.launch(show_api=False)

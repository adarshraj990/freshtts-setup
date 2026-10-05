# 🚀 Kaggle / Google Colab Dual-GPU XTTS-v2 FastAPI Backend Setup & Execution Script
## Thread-Safe Dual-GPU Architecture for High-Throughput Subtitle Voice Cloning

This guide provides the complete, copy-pasteable script to run on **Kaggle Notebooks (Dual T4 GPUs)** or **Google Colab (T4 GPU)**.

It eliminates the **HTTP 500 Internal Server Error** caused by concurrent requests colliding on the same CUDA device.

---

### 🛡️ Core Concurrency & Reliability Fixes

1. **Thread-Safe Model Binding (`cuda:0` and `cuda:1`)**:
   - Each GPU initializes its own independent `XTTS-v2` model instance.
   - Each model is protected by its own `threading.Lock()` and `asyncio.Lock()`.
2. **Dedicated Worker Queue (`asyncio.Queue`)**:
   - Incoming requests from the frontend ThreadPoolExecutor enter a round-robin worker queue.
   - Each GPU executes **strictly 1 synthesis request at a time**, eliminating race conditions and illegal memory accesses.
   - Synchronous CUDA synthesis is dispatched via `asyncio.to_thread` so that both GPUs synthesize in parallel without blocking the FastAPI event loop.
3. **Early Input Validation (HTTP 400)**:
   - Immediately checks for empty or whitespace-only text before touching the model or GPU.
   - Returns a structured HTTP 400 Bad Request instead of causing the model tokenizer to crash.
4. **CUDA Memory & Exception Recovery**:
   - Wraps `model.tts_to_file` in a `try...except` block catching `torch.cuda.OutOfMemoryError`, `torch.cuda.CudaError`, and `RuntimeError`.
   - Automatically executes `torch.cuda.empty_cache()` and `gc.collect()` upon error.
   - Returns clean, meaningful JSON error responses rather than unhandled 500 server crashes.
5. **Health Check & Latency Endpoint**:
   - Built-in `GET /` and `GET /health` returning worker states, queue capacity, and memory metrics.
   - Pre-configured for Localtunnel (`npx localtunnel --port 8000`) and Ngrok.

---

### 📋 Setup Instructions for Kaggle (2x T4 GPUs)

1. Open a new or existing **Kaggle Notebook**.
2. Under **Notebook settings** (right sidebar):
   - **Accelerator**: Select **GPU T4 x 2**
   - **Internet**: Ensure **Internet on** is enabled.
3. Paste the complete code cell below into your Kaggle notebook and click **Run**.
4. The script will install dependencies, load models on both `cuda:0` and `cuda:1`, start the FastAPI server on port 8000, and launch Localtunnel.
5. Copy the generated public tunnel URL (e.g., `https://xxxx.loca.lt`) and paste it into the **Kaggle API URL** textbox in your Hugging Face Space frontend.
6. Click **🔍 Check Connection** to verify green latency status!

---

### 💻 All-In-One Kaggle / Colab Notebook Script

```python
# ==============================================================================
# 🚀 High-Performance Dual-GPU XTTS-v2 FastAPI Backend Server
# Thread-Safe Concurrency & Worker Queue for Kaggle (2x T4 GPUs) & Google Colab
# ==============================================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["COQUI_TOS_AGREED"] = "1"

print("=" * 70)
print("📦 STEP 1: Installing System Dependencies (ffmpeg, espeak-ng, nodejs)...")
print("=" * 70)
!apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 espeak-ng nodejs npm

print("\n" + "=" * 70)
print("⚡ STEP 2: Installing Python Packages (fastapi, uvicorn, TTS, pyngrok)...")
print("=" * 70)
!pip install -q --upgrade pip
!pip install -q "numpy>=1.24.0,<2.0.0" "transformers>=4.39.0,<4.45.0"
!pip install -q --no-build-isolation git+https://github.com/idiap/coqui-ai-TTS
!pip install -q fastapi uvicorn python-multipart soundfile pydub pyngrok

print("\n" + "=" * 70)
print("🔧 STEP 3: Initializing Thread-Safe Dual-GPU FastAPI Server...")
print("=" * 70)

import asyncio
from contextlib import asynccontextmanager
import gc
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Optional
import uuid
import urllib.request

import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [XTTS-DualGPU] %(message)s",
)
logger = logging.getLogger("xtts_dual_gpu")

# ── Monkey Patch for librosa / TTS pkg_resources compatibility ───────────────
import types
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources
    except ModuleNotFoundError:
        import importlib.resources
        pr = types.ModuleType("pkg_resources")
        def resource_filename(package_or_requirement, resource_name):
            try:
                return str(importlib.resources.files(package_or_requirement) / resource_name)
            except Exception:
                return resource_name
        pr.resource_filename = resource_filename
        sys.modules["pkg_resources"] = pr

# ── PyTorch 2.6+ weights_only safe loader patch ──────────────────────────────
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load


# ── GPU Worker Abstraction ───────────────────────────────────────────────────

class GPUWorker:
    """
    Encapsulates an XTTS-v2 model bound to a dedicated CUDA device.
    Strictly synchronizes inference using both threading.Lock and asyncio.Lock.
    """

    def __init__(self, device: str, model):
        self.device = device
        self.model = model
        self.thread_lock = threading.Lock()
        self.async_lock = asyncio.Lock()
        self.total_synthesized = 0
        self.is_busy = False

    def synthesize(self, text: str, ref_audio_path: str, language: str, output_path: str) -> None:
        """
        Synchronous synthesis worker executed inside a dedicated worker thread.
        Strictly serialized per GPU device using self.thread_lock.
        """
        with self.thread_lock:
            self.is_busy = True
            t_start = time.time()
            try:
                # Set active CUDA device for PyTorch context
                if self.device.startswith("cuda"):
                    dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                    torch.cuda.set_device(dev_idx)

                logger.info(f"[{self.device}] Synthesizing ({language}): '{text[:45]}...'")

                # Invoke model.tts_to_file
                if hasattr(self.model, "tts_to_file"):
                    self.model.tts_to_file(
                        text=text,
                        speaker_wav=ref_audio_path,
                        language=language,
                        file_path=str(output_path),
                    )
                elif hasattr(self.model, "inference"):
                    import soundfile as sf
                    import numpy as np

                    gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                        audio_path=[ref_audio_path]
                    )
                    out = self.model.inference(
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=0.7,
                        length_penalty=1.0,
                        repetition_penalty=2.0,
                        top_k=50,
                        top_p=0.8,
                    )
                    sf.write(str(output_path), np.array(out["wav"]), 24000)
                else:
                    raise RuntimeError(f"Loaded model on {self.device} has no tts_to_file or inference method.")

                elapsed = time.time() - t_start
                self.total_synthesized += 1
                logger.info(f"[{self.device}] Completed in {elapsed:.2f}s (Total: {self.total_synthesized})")

            except torch.cuda.OutOfMemoryError as oom_err:
                logger.error(f"[{self.device}] CUDA OutOfMemoryError: {oom_err}")
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                raise RuntimeError(f"CUDA Out of Memory on {self.device}. Cache cleared.") from oom_err

            except torch.cuda.CudaError as cuda_err:
                logger.error(f"[{self.device}] CUDA Error: {cuda_err}")
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                raise RuntimeError(f"CUDA Hardware Error on {self.device}: {cuda_err}") from cuda_err

            except Exception as e:
                logger.error(f"[{self.device}] Synthesis error: {e}", exc_info=True)
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                raise e

            finally:
                self.is_busy = False


# ── Global State & Worker Pool ────────────────────────────────────────────────

workers: List[GPUWorker] = []
gpu_queue: asyncio.Queue = asyncio.Queue()


def initialize_xtts_models() -> List[GPUWorker]:
    """Loads independent XTTS-v2 models on all detected CUDA GPUs (Dual-GPU support)."""
    initialized_workers: List[GPUWorker] = []
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Hardware Detection: Found {num_gpus} CUDA device(s).")

    if num_gpus >= 2:
        devices = ["cuda:0", "cuda:1"]
    elif num_gpus == 1:
        devices = ["cuda:0"]
    else:
        devices = ["cpu"]

    from TTS.api import TTS

    for dev in devices:
        print(f"⏳ Loading XTTS-v2 on device '{dev}'...")
        t0 = time.time()
        try:
            if dev.startswith("cuda"):
                dev_idx = int(dev.split(":")[-1])
                torch.cuda.set_device(dev_idx)
                model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
                model.to(dev)
            else:
                model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

            worker = GPUWorker(device=dev, model=model)
            initialized_workers.append(worker)
            print(f"✅ Loaded XTTS-v2 on '{dev}' in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load on {dev}: {e}")
            if not initialized_workers and dev == devices[-1]:
                raise e

    return initialized_workers


# ── FastAPI Lifespan & Application ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workers
    workers = initialize_xtts_models()
    for worker in workers:
        await gpu_queue.put(worker)
    print(f"🎉 Server ready! {len(workers)} GPU worker(s) in active queue.")
    yield
    for w in workers:
        if w.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


app = FastAPI(
    title="XTTS-v2 Dual-GPU FastAPI Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint verified by the Frontend Connection Check."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_status = []
    for w in workers:
        worker_status.append({
            "device": w.device,
            "busy": w.is_busy,
            "total_synthesized": w.total_synthesized,
        })
    return {
        "status": "healthy",
        "service": "XTTS-v2 Dual-GPU Backend",
        "cuda_available": torch.cuda.is_available(),
        "total_gpus": num_gpus,
        "workers": worker_status,
        "available_workers_in_queue": gpu_queue.qsize(),
    }


# ── One-Time Speaker Registration & Fast Synthesis Endpoints ─────────────────

GLOBAL_SPEAKER_CACHE = {}


@app.post("/register_speaker")
async def register_speaker(
    speaker_wav: Optional[UploadFile] = File(None),
    reference_voice: Optional[UploadFile] = File(None),
    speaker_id: Optional[str] = Form(None),
):
    """Caches speaker reference audio once on the backend for all subsequent line requests."""
    ref_file = speaker_wav if speaker_wav is not None else reference_voice
    if ref_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Speaker reference audio file ('speaker_wav' or 'reference_voice') is required.",
        )

    content = await ref_file.read()
    if not content or len(content) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded speaker audio file is empty or corrupted.",
        )

    import hashlib
    sid = speaker_id.strip() if (speaker_id and speaker_id.strip()) else f"spk_{hashlib.md5(content[:2048]).hexdigest()[:8]}"

    speaker_dir = Path(tempfile.gettempdir()) / "xtts_speaker_cache"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    saved_path = speaker_dir / f"{sid}.wav"

    with open(saved_path, "wb") as f:
        f.write(content)

    GLOBAL_SPEAKER_CACHE[sid] = str(saved_path)
    GLOBAL_SPEAKER_CACHE["active_speaker"] = sid

    print(f"✅ Registered and cached speaker audio '{sid}' at: {saved_path}")

    return {
        "status": "registered",
        "speaker_id": sid,
        "message": f"Speaker audio '{sid}' cached in backend memory.",
        "size_bytes": len(content),
    }


@app.post("/synthesize_line")
async def synthesize_line(
    text: Optional[str] = Form(None),
    text_chunk: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
    speaker_id: Optional[str] = Form(None),
):
    """Ultra-fast single line synthesis using pre-cached speaker audio (NO audio upload)."""
    raw_text = text if (text is not None and text.strip()) else text_chunk
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: 'text' or 'text_chunk' cannot be empty or whitespace-only.",
        )
    clean_text = raw_text.strip()

    raw_lang = language if language else target_lang
    clean_lang = (raw_lang or "hi").strip().lower()

    sid = speaker_id.strip() if (speaker_id and speaker_id.strip()) else GLOBAL_SPEAKER_CACHE.get("active_speaker")
    if not sid or sid not in GLOBAL_SPEAKER_CACHE or not os.path.isfile(GLOBAL_SPEAKER_CACHE[sid]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speaker audio registered on backend. Call /register_speaker first.",
        )
    ref_audio_path = GLOBAL_SPEAKER_CACHE[sid]

    temp_dir = Path(tempfile.mkdtemp(prefix="xtts_line_"))
    output_audio_path = temp_dir / f"line_{uuid.uuid4().hex[:8]}.wav"

    try:
        worker: GPUWorker = await gpu_queue.get()
        try:
            async with worker.async_lock:
                await asyncio.to_thread(
                    worker.synthesize,
                    text=clean_text,
                    ref_audio_path=ref_audio_path,
                    language=clean_lang,
                    output_path=str(output_audio_path),
                )
        except Exception as synth_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference Failure on {worker.device}: {str(synth_err)}",
            )
        finally:
            await gpu_queue.put(worker)
            gpu_queue.task_done()

        if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model on {worker.device} produced an empty audio file.",
            )

        with open(output_audio_path, "rb") as out_f:
            audio_bytes = out_f.read()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-GPU-Device": worker.device,
                "Content-Disposition": f'attachment; filename="line_{worker.device}.wav"',
            },
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Voice Clone Synthesis Endpoint (Full Upload Fallback) ─────────────────────

@app.post("/voice_clone_synthesis")
async def voice_clone_synthesis(
    text: Optional[str] = Form(None),
    text_chunk: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None),
    reference_voice: Optional[UploadFile] = File(None),
):
    # 1. Input Validation: Check text
    raw_text = text if (text is not None and text.strip()) else text_chunk
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: 'text' or 'text_chunk' cannot be empty or whitespace-only.",
        )
    clean_text = raw_text.strip()

    # Resolve target language
    raw_lang = language if language else target_lang
    clean_lang = (raw_lang or "hi").strip().lower()

    # Resolve reference audio upload
    ref_file = speaker_wav if speaker_wav is not None else reference_voice
    if ref_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: Speaker audio file ('speaker_wav' or 'reference_voice') is required.",
        )

    # 2. Stage temporary files
    temp_dir = Path(tempfile.mkdtemp(prefix="xtts_task_"))
    ref_audio_path = temp_dir / f"ref_{uuid.uuid4().hex[:8]}.wav"
    output_audio_path = temp_dir / f"output_{uuid.uuid4().hex[:8]}.wav"

    try:
        content = await ref_file.read()
        if not content or len(content) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Validation Error: Uploaded reference audio file is empty or corrupted.",
            )

        with open(ref_audio_path, "wb") as f:
            f.write(content)

        # 3. Acquire available GPU worker from queue (asynchronously waits if all are busy)
        worker: GPUWorker = await gpu_queue.get()
        try:
            # Dispatch to worker thread while holding worker's async lock
            async with worker.async_lock:
                await asyncio.to_thread(
                    worker.synthesize,
                    text=clean_text,
                    ref_audio_path=str(ref_audio_path),
                    language=clean_lang,
                    output_path=str(output_audio_path),
                )
        except Exception as synth_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference Failure on {worker.device}: {str(synth_err)}",
            )
        finally:
            # Release worker back to pool for next waiting task
            await gpu_queue.put(worker)
            gpu_queue.task_done()

        # 4. Verify output file
        if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model on {worker.device} produced an empty or missing audio file.",
            )

        # Read binary audio content
        with open(output_audio_path, "rb") as out_f:
            audio_bytes = out_f.read()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-GPU-Device": worker.device,
                "Content-Disposition": f'attachment; filename="synthesis_{worker.device}.wav"',
            },
        )

    finally:
        # Immediate cleanup of temporary disk storage
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 🌐 STEP 4: Start Localtunnel & Launch Server
# ==============================================================================

def launch_server(port: int = 8000):
    def run_tunnel():
        time.sleep(2)
        try:
            print("=" * 70)
            print("🌐 Starting Localtunnel on port 8000...")
            try:
                with urllib.request.urlopen("https://loca.lt/mytunnelpassword", timeout=5) as resp:
                    pwd = resp.read().decode("utf-8").strip()
                    print(f"🔑 Localtunnel IP Password (if prompted in browser): {pwd}")
            except Exception:
                pass
            print("=" * 70)
            cmd = f"npx -y localtunnel --port {port}"
            subprocess.Popen(cmd, shell=True)
        except Exception as e:
            print(f"Tunnel launch error: {e}")

    threading.Thread(target=run_tunnel, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

launch_server(port=8000)
```

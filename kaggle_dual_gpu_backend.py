"""
🚀 High-Performance Dual-GPU XTTS-v2 FastAPI Backend Server
Designed for Kaggle (2x T4 GPUs), Google Colab, and Multi-GPU Environments.

Concurrency & Stability Architecture:
1. Dedicated GPU Worker Instances:
   - Separate XTTS-v2 instance loaded on 'cuda:0' and 'cuda:1' (or CPU fallback).
   - Dedicated threading.Lock() and asyncio.Lock() per GPU instance.
   - An asyncio.Queue worker pool distributes requests across available GPUs.
   - Guarantees each GPU processes strictly ONE synthesis task at a time (zero thread collision).
2. Non-Blocking Async Execution:
   - Dispatches synchronous PyTorch synthesis via asyncio.to_thread to run both GPUs in parallel.
3. Input Validation:
   - Validates text and reference audio; returns HTTP 400 for empty or whitespace-only text.
4. Error & CUDA Memory Management:
   - Catches CUDA OOM and runtime errors, invokes torch.cuda.empty_cache() and gc.collect().
   - Returns structured JSON error messages instead of raw 500 crashes.
5. Health Checks & Tunnel Support:
   - GET / and GET /health endpoints for instant latency & status monitoring.
   - Compatible with Localtunnel, Ngrok, and Cloudflare tunnels.
"""

import asyncio
from contextlib import asynccontextmanager
import gc
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [XTTS-DualGPU] %(message)s",
)
logger = logging.getLogger("xtts_dual_gpu")


# ── GPU Worker Abstraction ───────────────────────────────────────────────────

class GPUWorker:
    """
    Encapsulates a single XTTS-v2 model bound to a dedicated CUDA device.
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
        Strictly serialized per GPU using self.thread_lock.
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
                    # Fallback for direct Xtts model instances
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
                logger.info(f"[{self.device}] Synthesis finished in {elapsed:.2f}s (Total on GPU: {self.total_synthesized})")

            except torch.cuda.OutOfMemoryError as oom_err:
                logger.error(f"[{self.device}] CUDA OutOfMemoryError: {oom_err}")
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                raise RuntimeError(f"CUDA Out of Memory on {self.device}. Cache cleared. Try shortening text chunk.") from oom_err

            except torch.cuda.CudaError as cuda_err:
                logger.error(f"[{self.device}] CUDA Error: {cuda_err}")
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()
                raise RuntimeError(f"CUDA Hardware/Driver Error on {self.device}: {cuda_err}") from cuda_err

            except Exception as e:
                logger.error(f"[{self.device}] Synthesis failed: {e}", exc_info=True)
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
    """
    Initializes independent XTTS-v2 model instances on all detected CUDA GPUs.
    Supports Dual-GPU (cuda:0 and cuda:1 on Kaggle), single GPU, or CPU fallback.
    """
    initialized_workers: List[GPUWorker] = []

    # Monkey patch pkg_resources for librosa/TTS compatibility
    import sys
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

    # Safe PyTorch 2.6+ weights_only loader patch
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Hardware Detection: Found {num_gpus} CUDA device(s).")

    if num_gpus >= 2:
        devices = ["cuda:0", "cuda:1"]
    elif num_gpus == 1:
        devices = ["cuda:0"]
    else:
        devices = ["cpu"]

    from TTS.api import TTS

    for dev in devices:
        logger.info(f"⏳ Initializing XTTS-v2 on device '{dev}'...")
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
            logger.info(f"✅ Successfully loaded XTTS-v2 on '{dev}' in {time.time() - t0:.2f}s")
        except Exception as init_err:
            logger.error(f"❌ Failed to load model on {dev}: {init_err}", exc_info=True)
            if not initialized_workers and dev == devices[-1]:
                raise init_err

    return initialized_workers


# ── FastAPI Application Lifespan ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workers
    logger.info("🚀 Starting Dual-GPU XTTS-v2 FastAPI Backend Service...")
    workers = initialize_xtts_models()
    for worker in workers:
        await gpu_queue.put(worker)
    logger.info(f"🎉 Engine ready! {len(workers)} worker(s) active in saturated queue.")
    yield
    logger.info("🛑 Shutting down Dual-GPU XTTS-v2 Backend...")
    for w in workers:
        if w.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


app = FastAPI(
    title="XTTS-v2 Dual-GPU Distributed API Backend",
    description="High-throughput thread-safe FastAPI backend for subtitle dubbing and voice cloning.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check Endpoints ───────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
async def health_check():
    """
    Lightweight health check endpoint returning GPU status, worker availability,
    and memory statistics for the frontend connection health check.
    """
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

GLOBAL_SPEAKER_CACHE: Dict[str, str] = {}


@app.post("/register_speaker")
async def register_speaker(
    speaker_wav: Optional[UploadFile] = File(None),
    reference_voice: Optional[UploadFile] = File(None),
    speaker_id: Optional[str] = Form(None),
):
    """
    Caches speaker reference audio once on the backend.
    Enables all subsequent subtitle chunks to call /synthesize_line without re-uploading audio.
    """
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

    # Derive speaker ID
    import hashlib
    sid = speaker_id.strip() if (speaker_id and speaker_id.strip()) else f"spk_{hashlib.md5(content[:2048]).hexdigest()[:8]}"

    speaker_dir = Path(tempfile.gettempdir()) / "xtts_speaker_cache"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    saved_path = speaker_dir / f"{sid}.wav"

    with open(saved_path, "wb") as f:
        f.write(content)

    GLOBAL_SPEAKER_CACHE[sid] = str(saved_path)
    GLOBAL_SPEAKER_CACHE["active_speaker"] = sid

    logger.info(f"✅ Registered and cached speaker audio '{sid}' at: {saved_path}")

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
    """
    Ultra-fast single line synthesis using pre-cached speaker audio.
    Requires NO audio file upload over the network tunnel!
    """
    # 1. Input Validation
    raw_text = text if (text is not None and text.strip()) else text_chunk
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: 'text' or 'text_chunk' cannot be empty or whitespace-only.",
        )
    clean_text = raw_text.strip()

    raw_lang = language if language else target_lang
    clean_lang = (raw_lang or "hi").strip().lower()

    # Resolve cached speaker
    sid = speaker_id.strip() if (speaker_id and speaker_id.strip()) else GLOBAL_SPEAKER_CACHE.get("active_speaker")
    if not sid or sid not in GLOBAL_SPEAKER_CACHE or not os.path.isfile(GLOBAL_SPEAKER_CACHE[sid]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No speaker audio registered on backend. Call /register_speaker first.",
        )
    ref_audio_path = GLOBAL_SPEAKER_CACHE[sid]

    # 2. Stage temporary output
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
    """
    Dispatches synthesis to the next available GPU worker in the pool.
    Guarantees strict single-thread execution per GPU device.
    """
    # 1. Input Validation: Check text
    raw_text = text if (text is not None and text.strip()) else text_chunk
    if not raw_text or not raw_text.strip():
        logger.warning("Rejected synthesis request: empty or whitespace-only text.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: 'text' or 'text_chunk' cannot be empty or whitespace-only.",
        )
    clean_text = raw_text.strip()

    # Resolve target language code
    raw_lang = language if language else target_lang
    clean_lang = (raw_lang or "hi").strip().lower()

    # Resolve reference audio upload
    ref_file = speaker_wav if speaker_wav is not None else reference_voice
    if ref_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Validation Error: Speaker reference audio file ('speaker_wav' or 'reference_voice') is required.",
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

        # 3. Acquire available GPU worker from queue (blocks asynchronously until a GPU is free)
        worker: GPUWorker = await gpu_queue.get()
        try:
            # Concurrently synthesize using asyncio.to_thread while holding the worker's async lock
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
            # Always return worker to pool for next waiting task
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


# ── Tunnel & Server Execution Helpers ────────────────────────────────────────

def launch_server(port: int = 8000, use_localtunnel: bool = True, ngrok_token: Optional[str] = None):
    """
    Launches uvicorn server along with Localtunnel or Ngrok for instant public access.
    """
    import subprocess
    import threading

    def run_tunnel():
        time.sleep(2)
        if ngrok_token:
            try:
                from pyngrok import ngrok
                ngrok.set_auth_token(ngrok_token)
                public_url = ngrok.connect(port).public_url
                print("=" * 70)
                print(f"🌐 NGROK PUBLIC API URL: {public_url}")
                print(f"👉 Target Endpoint: {public_url}/voice_clone_synthesis")
                print("=" * 70)
                return
            except Exception as e:
                print(f"Ngrok launch failed: {e}. Falling back to Localtunnel.")

        if use_localtunnel:
            try:
                print("=" * 70)
                print("🌐 Launching Localtunnel (port 8000)...")
                # Retrieve Localtunnel password
                import urllib.request
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
                print(f"Localtunnel failed: {e}")

    threading.Thread(target=run_tunnel, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    launch_server(port=8000, use_localtunnel=True)

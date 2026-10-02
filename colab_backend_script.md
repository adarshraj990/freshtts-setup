# Google Colab GPU Backend Setup & Execution Script
## High-Fidelity XTTS-v2 Engine (Adapted from Bhojpuri XTTS API)

This document contains the exact code to run on **Google Colab** with a **T4 GPU** (free tier). 
The architecture directly extracts and adapts the core model loading, monkey patches, and latent caching logic from the high-quality **Bhojpuri XTTS API Space** (`https://huggingface.co/spaces/Aviranjanprasad/Bhojpuri-XTTS-API`), scaling it for multi-language dubbing (**Hindi, French, Spanish, Portuguese**).

---

### 📋 Setup Instructions for Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Select **Runtime** → **Change runtime type** → select **T4 GPU** → click **Save**.
3. Copy the complete code cell below into Google Colab and click **Run (Shift + Enter)**.
4. When finished launching, copy the generated public link (`https://xxxxxxxx.gradio.live`) and paste it into **Google Colab API URL 1** in your Frontend UI.

---

### 💻 Colab All-In-One Script

```python
# ==============================================================================
# 🚀 Coqui XTTS-v2 GPU Backend Server
# Adapted from Bhojpuri XTTS API (Aviranjanprasad/Bhojpuri-XTTS-API)
# Exposes a Public Gradio API Endpoint for the Hugging Face Frontend
# ==============================================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["COQUI_TOS_AGREED"] = "1"

print("=" * 70)
print("📦 STEP 1: Installing System Dependencies (ffmpeg, espeak-ng, libsndfile1)...")
print("=" * 70)
!apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 espeak-ng

print("\n" + "=" * 70)
print("⚡ STEP 2: Installing Core Python Packages (TTS, Gradio, Pydub, Soundfile)...")
print("=" * 70)
!pip install -q --upgrade pip
!pip install -q "numpy>=1.24.0,<2.0.0" "transformers>=4.39.0,<4.45.0"
!pip install -q --no-build-isolation git+https://github.com/idiap/coqui-ai-TTS
!pip install -q gradio pydub soundfile huggingface_hub

print("\n" + "=" * 70)
print("🔧 STEP 3: Applying Bhojpuri XTTS Patches & Loading XTTS-v2 Model...")
print("=" * 70)

import sys
import types
import time
import uuid
import tempfile
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
import gradio as gr
from huggingface_hub import snapshot_download

# ── 1. Monkey-patch broken pkg_resources for librosa / TTS ────────────────────
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources
    except ModuleNotFoundError:
        import importlib.resources
        pr = types.ModuleType("pkg_resources")
        def resource_filename(package_or_requirement, resource_name):
            try:
                import importlib.resources as ir
                return str(ir.files(package_or_requirement) / resource_name)
            except Exception:
                return resource_name
        pr.resource_filename = resource_filename
        sys.modules["pkg_resources"] = pr

# ── 2. Monkey-patch torchaudio.load to use soundfile (bypasses torchcodec) ───
def _safe_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T)
    return tensor, sr

torchaudio.load = _safe_torchaudio_load

# ── 3. PyTorch 2.6 weights_only safe loader patch ─────────────────────────────
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _safe_torch_load

# ── 4. Verify GPU/CUDA Acceleration ──────────────────────────────────────────
if not torch.cuda.is_available():
    print("⚠️ WARNING: CUDA not detected! Running on CPU. For fast dubbing, enable T4 GPU.")
    device = "cpu"
else:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ CUDA GPU Active: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    device = "cuda"

# ── 5. Download and Load XTTS-v2 Checkpoint (Bhojpuri Space Architecture) ────
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

t_start = time.time()
print("Downloading / loading XTTS-v2 weights from Hugging Face Hub...")
checkpoint_dir = snapshot_download(repo_id="coqui/XTTS-v2")

config = XttsConfig()
config.load_json(os.path.join(checkpoint_dir, "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)

# Extend Hindi tokenizer character limits if present
if "hi" not in model.tokenizer.char_limits:
    model.tokenizer.char_limits["hi"] = 250

if device == "cuda":
    model.cuda()

print(f"🔥 XTTS-v2 Model successfully initialized on {device.upper()} in {time.time() - t_start:.2f}s!")

# ── 6. Speaker Conditioning Latent Cache ─────────────────────────────────────
speaker_latents_cache = {}

def get_speaker_latents(audio_path: str):
    """Computes or retrieves cached speaker conditioning latents for zero-shot voice cloning."""
    cache_key = str(Path(audio_path).resolve())
    if cache_key in speaker_latents_cache:
        return speaker_latents_cache[cache_key]

    print(f"[Latent Cache] Computing conditioning latents for: {Path(audio_path).name}...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(audio_path)])
    latents = {
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }
    speaker_latents_cache[cache_key] = latents
    return latents

# Temporary output directory for generated dialogue clips
TEMP_OUTPUT_DIR = Path(tempfile.gettempdir()) / "xtts_chunks"
TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 7. Inference Function ────────────────────────────────────────────────────
def synthesize_xtts(text_chunk: str, target_lang: str, reference_voice) -> str:
    """
    Synthesizes speech using the core Bhojpuri XTTS-v2 logic.
    CRITICAL: Does NOT alter or clean the text in any way before passing to model.
    """
    if not text_chunk:
        raise gr.Error("text_chunk cannot be empty.")
        
    if not reference_voice:
        raise gr.Error("reference_voice audio file is required.")

    # Resolve reference voice filepath
    if isinstance(reference_voice, str):
        ref_path = reference_voice
    elif hasattr(reference_voice, "name"):
        ref_path = reference_voice.name
    else:
        ref_path = str(reference_voice)

    if not os.path.exists(ref_path):
        raise gr.Error(f"Reference voice not found at: {ref_path}")

    # Retrieve speaker latents
    latents = get_speaker_latents(ref_path)
    lang_code = target_lang.strip().lower()

    print(f"🎙️ Synthesizing [{lang_code.upper()}]: '{text_chunk[:35]}...'")

    # Core inference using Bhojpuri XTTS sampling hyperparameters
    out = model.inference(
        text=text_chunk,  # Passed EXACTLY as received from SRT
        language=lang_code,
        gpt_cond_latent=latents["gpt_cond_latent"],
        speaker_embedding=latents["speaker_embedding"],
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.8,
        speed=1.0,
        enable_text_splitting=True
    )

    wav = np.array(out["wav"])
    chunk_filename = f"chunk_{uuid.uuid4().hex[:10]}_{lang_code}.wav"
    output_path = TEMP_OUTPUT_DIR / chunk_filename

    # Save 24kHz audio via soundfile
    sf.write(str(output_path), wav, 24000)
    return str(output_path)

# ==============================================================================
# 🌐 STEP 4: Launch Public Gradio Interface (share=True)
# ==============================================================================
demo = gr.Interface(
    fn=synthesize_xtts,
    inputs=[
        gr.Textbox(label="text_chunk", placeholder="Dialogue line exactly from SRT..."),
        gr.Textbox(label="target_lang", placeholder="hi, fr, es, pt"),
        gr.File(label="reference_voice", type="filepath")
    ],
    outputs=gr.Audio(label="Synthesized Audio (.wav)", type="filepath"),
    title="⚡ Bhojpuri-Adapted Coqui XTTS-v2 Multi-Language Backend",
    description="Dedicated GPU execution endpoint for the Hybrid Anime Dubbing Pipeline."
)

print("\n" + "=" * 70)
print("🚀 Launching Gradio API with Public Share Link (share=True)...")
print("=" * 70)

demo.launch(share=True, debug=False)
```

---

### ⚖️ Multi-GPU Parallel Dubbing
To achieve $2\times$ or $3\times$ dubbing speed:
1. Open up to 3 Google Colab notebooks under separate tabs/accounts.
2. Run this identical script in each notebook.
3. Paste each `.gradio.live` link into **URL 1**, **URL 2**, and **URL 3** on your Frontend.
4. The frontend will automatically load-balance subtitle chunks in a **Round-Robin** sequence!

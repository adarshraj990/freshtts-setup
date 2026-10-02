---
title: Hybrid Anime Dubbing Pipeline
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🎬 Hybrid Anime Dubbing Pipeline (SRT Lip-Sync & Distributed XTTS-v2)
### High-Performance Multilingual Voice Dubbing via Hugging Face Space & Google Colab GPU Cluster

[![Model: Coqui XTTS-v2](https://img.shields.io/badge/Model-Coqui%20XTTS--v2-6366f1.svg)](https://huggingface.co/coqui/XTTS-v2)
[![UI: Gradio Blocks](https://img.shields.io/badge/Frontend-Gradio%20Blocks-ff4b4b.svg)](https://gradio.app)
[![Architecture: Hybrid Decoupled](https://img.shields.io/badge/Architecture-Hybrid%20Frontend--Backend-10b981.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## 📖 1. What is the Hybrid Architecture?

Running deep voice cloning models like **Coqui XTTS-v2** directly on serverless community platforms (such as free Hugging Face Spaces with ZeroGPU) frequently leads to:
- ❌ **CUDA Out-of-Memory (OOM) fatal crashes** when processing consecutive dialogue blocks.
- ❌ **ZeroGPU quota exhaustion and aggressive timeouts** that terminate dubbing jobs halfway through an anime episode.
- ❌ **Sluggish processing** from repeated uploads of 20–30MB voice samples on every chunk.

To solve this, our **Hybrid Anime Dubbing Pipeline** completely decouples the application into two specialized tiers:

```
┌─────────────────────────────────────────────────────────┐
│        HUGGING FACE SPACE (Frontend / CPU Node)         │
│  • Gradio Blocks UI                                     │
│  • Subtitle Parsing (pysrt with millisecond accuracy)   │
│  • Round-Robin Multi-GPU Load Balancer                  │
│  • Master Timeline Audio Canvas Overlay (pydub)         │
│  • Hardcoded Reference Voice (reference_voice.wav)      │
└────────────────────────────┬────────────────────────────┘
                             │
            Round-Robin API Requests (gradio_client)
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  COLAB WORKER 1  │    │  COLAB WORKER 2  │    │  COLAB WORKER 3  │
│  (Google Colab)  │    │    (Optional)    │    │    (Optional)    │
│  • T4 GPU (15GB) │    │  • T4 GPU (15GB) │    │  • T4 GPU (15GB) │
│  • Coqui XTTS-v2 │    │  • Coqui XTTS-v2 │    │  • Coqui XTTS-v2 │
│  • Gradio API    │    │  • Gradio API    │    │  • Gradio API    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

1. **Frontend (Hugging Face Space or Local)**:
   - Lightweight, zero-GPU requirement (runs smoothly on free CPU tier).
   - Reads your local `reference_voice.wav` once from the root directory.
   - Parses `.srt` subtitle timing blocks down to the millisecond using `pysrt`.
   - Distributes dialogue chunks across up to **3 Google Colab GPU backends** in a **Round-Robin** sequence.
   - Combines returned audio segments onto a timeline canvas using `pydub.AudioSegment.overlay()` and exports `final_dubbed_output.wav`.
2. **Backend (Google Colab T4 GPU Cluster)**:
   - Executes the heavy Coqui XTTS-v2 neural model on a dedicated NVIDIA T4 GPU (15 GB VRAM).
   - Exposes a fast, authenticated Gradio API via a public share link (`https://xxxxxxxx.gradio.live`).

---

## 🎙️ 2. Step 1: Place Your Reference Voice in the Root Directory

To eliminate repetitive uploads across requests and ensure voice consistency, place your character reference audio file directly in the project root:

```text
audiogenflow/
├── reference_voice.wav    <--- PLACE YOUR CHARACTER VOICE SAMPLE HERE (.wav)
├── app.py
├── requirements.txt
├── PRD.md
├── colab_backend_script.md
└── README.md
```

> **Requirements for `reference_voice.wav`**:
> - Format: Standard uncompressed PCM `.wav` (mono or stereo, 22kHz–48kHz).
> - Content: Clean character voice with minimal background noise or background BGM (e.g. Naruto's voice dialogue).
> - Length: 5 seconds to 30 seconds works best for zero-shot speaker embedding extraction.

---

## 🚀 3. Step 2: Launch the Google Colab GPU Backend

1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Set the runtime to GPU: **Runtime** → **Change runtime type** → select **T4 GPU** → click **Save**.
3. Copy the full script from [colab_backend_script.md](file:///c:/Users/Adarsh/Desktop/audiogenflow/colab_backend_script.md) into a single Colab cell and press **Shift + Enter**.
4. The script will:
   - Automatically install `ffmpeg`, `espeak-ng`, `libsndfile1`, and `coqui-tts`.
   - Load `tts_models/multilingual/multi-dataset/xtts_v2` directly into CUDA VRAM.
   - Output a public shareable URL, for example:
     ```text
     Running on local URL:  http://127.0.0.1:7860
     Running on public URL: https://3b2a19c4d5e6f7a8.gradio.live
     ```
5. **Copy this `.gradio.live` link**.

---

## ⚖️ 4. Optional: Multi-GPU Load Balancing (2x–3x Speedup)

Want to dub an entire 20-minute episode in a fraction of the time?
1. Open a second (or third) Google Colab notebook under another browser tab or Google account.
2. Run the same [colab_backend_script.md](file:///c:/Users/Adarsh/Desktop/audiogenflow/colab_backend_script.md) on each instance.
3. You will receive multiple public URLs:
   - `URL 1`: `https://worker1.gradio.live`
   - `URL 2`: `https://worker2.gradio.live`
   - `URL 3`: `https://worker3.gradio.live`
4. The frontend will automatically cycle through the URLs using **Round-Robin** selection (`urls[i % len(urls)]`) to synthesize multiple chunks concurrently!

---

## 🖥️ 5. Step 3: Run the Frontend Application

### Option A: Deploy to Hugging Face Spaces (Recommended)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with **Gradio SDK** on the free **CPU tier**.
2. Push or upload `app.py`, `requirements.txt`, and your `reference_voice.wav` to your Hugging Face Space repository.
3. Hugging Face will automatically install dependencies and launch your web studio!

### Option B: Run Locally
```bash
# 1. Install frontend dependencies
pip install -r requirements.txt

# 2. Launch the Gradio Studio
python app.py
```
Open **`http://localhost:7860`** in your browser.

---

## 🎬 6. Step 4: Dubbing Your Anime Subtitles

1. In the Web UI, paste your Colab URL into **Google Colab API URL 1 (Mandatory Primary)**.
   - If running multiple Colabs, paste URL 2 and URL 3 into the optional fields.
2. Upload your anime subtitle file (`.srt`).
3. Select your **Target Language**:
   - `hi` (Hindi)
   - `fr` (French)
   - `es` (Spanish)
   - `pt` (Portuguese)
4. Click **🎬 Start Dubbing**.
5. Watch the live progress and real-time execution logs.
6. Once complete, listen to your synced master track in the browser audio player and download **`final_dubbed_output.wav`**!

---

## ⏱️ 7. Subtitle Lip-Sync & Timeline Alignment Logic

```python
# 1. Parse SRT and calculate canvas length (End of last subtitle + 2000ms buffer)
subs = pysrt.open(srt_file_path)
total_duration_ms = subs[-1].end.ordinal + 2000

# 2. Create blank silent canvas
canvas = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)

# 3. For each dialogue line, overlay at exact visual start timestamp
for i, sub in enumerate(subs):
    start_time_ms = sub.start.ordinal
    chunk_audio = fetch_chunk_from_colab(sub.text, target_lang, ref_voice)
    canvas = canvas.overlay(chunk_audio, position=start_time_ms)

# 4. Export finished master track
canvas.export("final_dubbed_output.wav", format="wav")
```

---

## 📂 Project Structure

```text
audiogenflow/
├── PRD.md                  # Comprehensive Product Requirements Document
├── README.md               # Architecture and setup guide (this file)
├── requirements.txt        # Frontend dependencies (gradio, gradio_client, pysrt, pydub)
├── colab_backend_script.md # Exact script to run in Google Colab T4 GPU
├── app.py                  # Hugging Face Space Gradio frontend application
└── reference_voice.wav     # Character reference voice sample in root directory
```

---

## 📜 License
Released under the MIT License.

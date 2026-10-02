# Product Requirements Document (PRD) & Architecture Progress
## Project: Hybrid Anime Dubbing Pipeline (SRT Lip-Sync & Distributed XTTS-v2)

---

### 1. Document Control & Current Progress

- **Status**: Architecture Finalized & Implemented
- **Author**: Systems Architect & Technical Product Manager
- **Core Technology**: Coqui XTTS-v2 (Adapted from [Bhojpuri-XTTS-API](https://huggingface.co/spaces/Aviranjanprasad/Bhojpuri-XTTS-API)), Gradio Client, Pysrt, Pydub
- **Target Deployment**: Hugging Face Spaces (Frontend) + Google Colab T4 GPU Cluster (Backend)

#### Current Progress Update:
We have finalized the system architecture. To eliminate synthetic robotic artifacts and ensure world-class voice cloning, we have adopted and adapted the core neural loading and latent conditioning logic from the **Bhojpuri XTTS API Space** (`Aviranjanprasad/Bhojpuri-XTTS-API`). While the original space was constrained to Bhojpuri/Hindi single-turn synthesis with presets, we adapted its direct `Xtts` model loading, `weights_only=False` safety patch, `torchaudio.load` soundfile bridge, and speaker latent caching into a distributed, multi-language anime dubbing engine supporting **Hindi (`hi`)**, **French (`fr`)**, **Spanish (`es`)**, and **Portuguese (`pt`)**.

---

### 2. The Problem Statement & Hybrid Solution

#### Bottlenecks with Monolithic Web Space Deployments:
Running deep generative speech cloning models like **Coqui XTTS-v2** directly inside free Hugging Face Spaces (ZeroGPU) introduces fatal production hurdles:
1. **ZeroGPU Quotas & Aggressive Timeouts**: Video dubbing requires processing dozens of sequential subtitle dialogue blocks. ZeroGPU instances enforce strict request timeouts (60–120s max), killing long dubbing tasks prematurely.
2. **CUDA Out-of-Memory (OOM)**: Long inference sessions accumulate GPU memory fragments, causing unexpected worker terminations.
3. **Repeated Voice Upload Latency**: Uploading 10–30MB voice samples on every chunk introduces massive network overhead.

#### The "Hybrid Frontend-Backend" Architecture:
To solve this systematically, the pipeline is decoupled into two independent, purpose-built layers:

```
┌──────────────────────────────────────────────────────────────┐
│          HUGGING FACE SPACE (Frontend / CPU Node)            │
│  • Lightweight Gradio Blocks UI (No GPU / ZeroGPU required)  │
│  • Hardcoded Reference Voice (reference_voice.wav in root)   │
│  • Subtitle Parsing (pysrt with exact millisecond onsets)    │
│  • Round-Robin Multi-GPU Load Balancer (up to 3 endpoints)   │
│  • Pydub Silent Canvas Timeline Alignment & WAV Export       │
└──────────────────────────────┬───────────────────────────────┘
                               │
              Distributed API Calls (gradio_client)
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  COLAB WORKER 1    │   │  COLAB WORKER 2    │   │  COLAB WORKER 3    │
│  (Google Colab T4) │   │    (Optional)      │   │    (Optional)      │
│  • Adapted Bhojpuri│   │  • Adapted Bhojpuri│   │  • Adapted Bhojpuri│
│    XTTS-v2 Engine  │   │    XTTS-v2 Engine  │   │    XTTS-v2 Engine  │
│  • Latent Caching  │   │  • Latent Caching  │   │  • Latent Caching  │
│  • Gradio /predict │   │  • Gradio /predict │   │  • Gradio /predict │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

1. **Frontend (Hugging Face Space)**:
   - Hosted on Hugging Face Spaces (Standard free CPU tier).
   - Manages user interaction, `.srt` parsing, timeline calculations, and distributed API routing.
   - Leverages `reference_voice.wav` stored permanently in the root directory (zero upload UI clutter).
   - Generates the master silent canvas with `pydub`, overlays each dialogue chunk at its exact start timestamp, and exports `final_dubbed_output.wav`.
2. **Backend (Google Colab T4 GPU Cluster)**:
   - Runs on free NVIDIA T4 GPUs (15 GB VRAM) on Google Colab.
   - Incorporates the core model initialization from `Aviranjanprasad/Bhojpuri-XTTS-API`.
   - Computes conditioning latents once per reference voice and caches them for zero-latency consecutive inference.
   - Exposes a fast Gradio API (`/predict`) over a secure public share link (`https://xxxx.gradio.live`).

---

### 3. Key Features & Specifications

#### 3.1 Hardcoded Reference Voice (`reference_voice.wav`)
- **Specification**: A single voice sample named `reference_voice.wav` is placed in the project root directory.
- **UI Impact**: There is **no voice upload component** in the frontend UI. The frontend automatically validates `reference_voice.wav` on startup and sends it dynamically to the backend API.
- **Benefit**: Zero upload friction for users and 100% vocal consistency across all subtitle lines.

#### 3.2 Time-Based SRT Chunking for Lip-Sync Alignment
- **Specification**: Subtitles are parsed using `pysrt`.
- **Timing Extraction**: For each subtitle block $i$, the start time in milliseconds is extracted:
  $$\text{start\_time\_ms} = \text{sub.start.ordinal}$$
- **Raw Text Integrity**: Subtitle text is passed **exactly as it appears in the SRT** to the TTS engine without destructive modifications or truncation.
- **Canvas Generation**: The master audio canvas is initialized with duration:
  $$\text{Canvas Duration} = \text{last\_sub.end.ordinal} + 2000\text{ ms buffer}$$
- **Overlay**: Synthesized `.wav` chunks are layered on top of the silent canvas at their exact visual onset using `pydub.AudioSegment.overlay(position=start_time_ms)`.

#### 3.3 Round-Robin Load Balancing (Up to 3 Colab URLs)
- **Specification**: The frontend provides 3 API URL inputs:
  - `URL 1`: Mandatory primary Colab endpoint.
  - `URL 2`: Optional secondary Colab endpoint.
  - `URL 3`: Optional tertiary Colab endpoint.
- **Dispatch Algorithm**:
  $$\text{Assigned Endpoint} = \text{ActiveURLs}[i \pmod{N}]$$
  where $i$ is the subtitle index ($0, 1, 2, \dots$) and $N$ is the number of active URLs provided ($1 \le N \le 3$).
- **Network Resilience**: If a call times out or encounters network jitter, the client logs the warning and automatically retries the chunk on an alternate available worker.

---

### 4. Core Backend Model Logic (Adapted from Bhojpuri XTTS API)

The Colab backend replicates the architectural foundation of `Aviranjanprasad/Bhojpuri-XTTS-API`:

1. **`torchaudio.load` Soundfile Patch**: Bypasses `torchcodec` backend failures in modern PyTorch releases by redirecting `torchaudio.load` directly through `soundfile.read`.
2. **PyTorch 2.6+ `weights_only=False` Bridge**: Safely intercepts `torch.load` calls to allow complex Coqui XTTS model serialization structures to load seamlessly.
3. **`snapshot_download` Direct Checkpoint Loading**: Downloads `coqui/XTTS-v2` directly from Hugging Face Hub, loading weights via `XttsConfig` and `Xtts.init_from_config(config)`.
4. **Speaker Latent Caching (`get_conditioning_latents`)**: Caches `gpt_cond_latent` and `speaker_embedding` by file path hash, eliminating repeated embedding extraction for consecutive lines.
5. **High-Fidelity Generation Parameters**:
   - `temperature=0.7`
   - `length_penalty=1.0`
   - `repetition_penalty=2.0`
   - `top_k=50`
   - `top_p=0.8`
   - `speed=1.0`
   - `enable_text_splitting=True`

---

### 5. Multi-Language Scope

| Code | Target Language | Model Support |
|---|---|---|
| **`hi`** | Hindi | Native Coqui XTTS-v2 + Bhojpuri Space optimization |
| **`fr`** | French | Native Coqui XTTS-v2 multilingual |
| **`es`** | Spanish | Native Coqui XTTS-v2 multilingual |
| **`pt`** | Portuguese | Native Coqui XTTS-v2 multilingual |

---

### 6. File Deliverables Summary

1. `PRD.md`: This comprehensive specifications and progress document.
2. `README.md`: Operational guide detailing root voice placement and Colab setup.
3. `requirements.txt`: Minimal frontend requirements (`gradio`, `gradio_client`, `pysrt`, `pydub`).
4. `colab_backend_script.md`: Exact Python script replicating Bhojpuri XTTS logic for Colab T4 GPU.
5. `app.py`: Clean Hugging Face Space Gradio frontend application.

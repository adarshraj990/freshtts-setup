import base64
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, Tuple

import gradio as gr
import requests

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


def synthesize_speech(
    api_url: str,
    text: str,
    language: str,
    audio_path: Optional[str],
) -> Tuple[Optional[str], str]:
    """
    Connects to the external Kaggle FastAPI backend running Coqui XTTS-v2.
    Formats the endpoint, applies Localtunnel headers, and handles errors.
    """
    # 1. Validation
    if not api_url or not api_url.strip():
        return None, "❌ Error: Please enter your active Kaggle API URL."

    if not text or not text.strip():
        return None, "❌ Error: Please enter the text you want to synthesize."

    if not audio_path or not os.path.isfile(audio_path):
        return None, "❌ Error: Please upload or record a reference voice audio sample."

    # 2. Endpoint Formatting (Sanitize trailing slashes and ensure route)
    clean_url = api_url.strip().rstrip("/")
    if not clean_url.endswith("/voice_clone_synthesis"):
        endpoint = f"{clean_url}/voice_clone_synthesis"
    else:
        endpoint = clean_url

    # 3. Headers (Localtunnel & Ngrok bypass headers)
    headers = {
        "Bypass-Tunnel-Reminder": "true",
        "bypass-tunnel-reminder": "true",
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "AudioGenFlow-Frontend/1.0",
    }

    # 4. Form Data & File Preparation
    data = {
        "text": text.strip(),
        "text_chunk": text.strip(),
        "language": language.strip().lower(),
        "target_lang": language.strip().lower(),
    }

    filename = Path(audio_path).name
    mime_type = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"

    start_time = time.time()

    # 5. Remote API Call with Robust Error Handling
    try:
        with open(audio_path, "rb") as f1, open(audio_path, "rb") as f2:
            files = [
                ("speaker_wav", (filename, f1, mime_type)),
                ("reference_voice", (filename, f2, mime_type)),
            ]
            resp = requests.post(
                endpoint,
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        elapsed = time.time() - start_time

        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "").lower()

            # Create a dedicated temp file for the returned audio
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            out_filepath = temp_output.name
            temp_output.close()

            # Handle JSON response containing base64 audio
            if "application/json" in content_type:
                try:
                    payload = resp.json()
                    b64_audio = payload.get("audio_base64") or payload.get("audio")
                    if b64_audio:
                        with open(out_filepath, "wb") as out_f:
                            out_f.write(base64.b64decode(b64_audio))
                        return out_filepath, f"✅ Success: Audio generated in {elapsed:.2f}s via {endpoint}"
                except Exception as json_err:
                    return None, f"❌ JSON Parsing Error: Failed to decode audio payload ({json_err})."

            # Direct audio binary stream (WAV/MP3/Octet-Stream)
            with open(out_filepath, "wb") as out_f:
                out_f.write(resp.content)

            file_size_kb = os.path.getsize(out_filepath) / 1024
            return out_filepath, f"✅ Success: Received {file_size_kb:.1f} KB audio in {elapsed:.2f}s from {endpoint}"

        elif resp.status_code in [502, 503, 504]:
            return None, (
                f"❌ Gateway Error (HTTP {resp.status_code}): The tunnel is unreachable. "
                "Ensure your Kaggle notebook server is running and the tunnel URL is active."
            )
        elif resp.status_code == 404:
            return None, f"❌ Not Found (HTTP 404): Endpoint '{endpoint}' does not exist on the remote server."
        elif resp.status_code == 422:
            return None, f"❌ Unprocessable Entity (HTTP 422): Backend rejected payload parameters.\nDetails: {resp.text[:300]}"
        else:
            return None, f"❌ Backend Error (HTTP {resp.status_code}):\n{resp.text[:300]}"

    except requests.exceptions.Timeout:
        return None, "❌ Timeout Error: The remote Kaggle backend did not respond within 120 seconds."
    except requests.exceptions.ConnectionError:
        return None, (
            f"❌ Connection Error: Unable to connect to '{clean_url}'. "
            "Please check that your Localtunnel or Ngrok URL is active."
        )
    except Exception as e:
        return None, f"❌ Unexpected Error: {str(e)}"


# ── Gradio User Interface ─────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="🎙️ XTTS-v2 Voice Cloning Frontend") as demo:
        gr.Markdown(
            """
            # 🎙️ XTTS-v2 Voice Cloning Frontend
            ### Connects to an external Kaggle / GPU backend running Coqui XTTS-v2.
            """
        )

        with gr.Row():
            # LEFT COLUMN: User Inputs
            with gr.Column(scale=5):
                api_url_input = gr.Textbox(
                    label="Kaggle API URL",
                    placeholder="https://famous-sheep-wash.loca.lt or https://xxxx.ngrok-free.app",
                    value="",
                    lines=1,
                    info="Paste your active Localtunnel or Ngrok tunnel URL (never hardcoded).",
                )

                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Type or paste the text you want to synthesize...",
                    lines=4,
                )

                language_dropdown = gr.Dropdown(
                    label="Language Code",
                    choices=["hi", "en", "es", "fr", "de", "it", "pt", "ru", "ar", "ja", "ko"],
                    value="hi",
                    info="Select target speech language.",
                )

                ref_audio_input = gr.Audio(
                    label="Reference Voice Sample",
                    type="filepath",
                )

                submit_btn = gr.Button("🚀 Generate Speech", variant="primary", size="lg")

            # RIGHT COLUMN: Outputs
            with gr.Column(scale=5):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                )

                status_output = gr.Textbox(
                    label="Status / Error Log",
                    interactive=False,
                    lines=4,
                    placeholder="Operation status and errors will appear here...",
                )

        submit_btn.click(
            fn=synthesize_speech,
            inputs=[
                api_url_input,
                text_input,
                language_dropdown,
                ref_audio_input,
            ],
            outputs=[
                audio_output,
                status_output,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    try:
        app.launch(server_name="0.0.0.0", server_port=7860)
    except ValueError:
        app.launch()

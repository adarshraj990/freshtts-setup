"""
Utility to generate a dummy voice sample (WAV) for instant testing when no microphone or external voice file is available.
Generates a warm 24kHz harmonic waveform speech-like pulse sequence.
"""

import math
from pathlib import Path
import struct
import wave

SAMPLE_RATE = 24000
DURATION_SEC = 4.0
OUTPUT_PATH = Path(__file__).resolve().parent / "sample_voice.wav"


def generate_speech_like_wave(output_path: Path = OUTPUT_PATH) -> Path:
    """Generates a clean 24kHz 16-bit mono wav file with harmonic speech-like formant sweeps."""
    num_samples = int(SAMPLE_RATE * DURATION_SEC)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)

        raw_frames = bytearray()
        f0 = 150.0  # Fundamental frequency (voice pitch)

        for i in range(num_samples):
            t = i / SAMPLE_RATE
            # Modulate pitch slightly for natural cadence
            pitch = f0 + 25.0 * math.sin(2 * math.pi * 1.5 * t)
            # Harmonic formants (F1 ~ 500Hz, F2 ~ 1500Hz, F3 ~ 2500Hz)
            sample = (
                0.5 * math.sin(2 * math.pi * pitch * t)
                + 0.3 * math.sin(2 * math.pi * (pitch * 3) * t)
                + 0.15 * math.sin(2 * math.pi * (pitch * 5) * t)
            )
            # Amplitude envelope (fade in / out)
            env = min(1.0, t / 0.1) * min(1.0, (DURATION_SEC - t) / 0.2)
            sample *= env

            # 16-bit PCM integer scaling
            int_val = int(max(-32767, min(32767, sample * 24000)))
            raw_frames.extend(struct.pack("<h", int_val))

        wav_file.writeframes(raw_frames)

    return output_path


if __name__ == "__main__":
    generated_file = generate_speech_like_wave()
    print(f"Generated sample voice wav at: {generated_file}")

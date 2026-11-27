import os
import tempfile
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice


class TextToSpeechService:
    def __init__(self, model_path: str = "models/piper/en_US-lessac-medium.onnx"):
        """Load a Piper voice model from its local ONNX file."""
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Piper voice model not found: {self.model_path}. "
                "Download one with: python -m piper.download_voices "
                "--data-dir models/piper en_US-lessac-medium"
            )

        self.model = PiperVoice.load(str(self.model_path))

    def _write_wave(self, text: str, output_path: str | Path) -> None:
        with wave.open(str(output_path), "wb") as wav_file:
            self.model.synthesize_wav(text, wav_file)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize text and return its sample rate and normalized audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            self._write_wave(text, temp_path)
            with wave.open(str(temp_path), "rb") as wav_file:
                if wav_file.getsampwidth() != 2:
                    raise ValueError("Piper returned audio with an unsupported sample width")

                audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                channels = wav_file.getnchannels()
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                return wav_file.getframerate(), audio
        finally:
            os.unlink(temp_path)

    def save_voice_sample(self, text: str, output_path: str) -> None:
        """Save synthesized speech as a WAV file."""
        self._write_wave(text, output_path)

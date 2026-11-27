import importlib
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakePiperVoice:
    @classmethod
    def load(cls, model_path):
        return cls()

    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(np.array([0, 16384, -16384], dtype=np.int16).tobytes())


class TestTextToSpeechService(unittest.TestCase):
    def _import_service(self):
        with patch.dict("sys.modules", {"piper": SimpleNamespace(PiperVoice=FakePiperVoice)}):
            sys.modules.pop("tts", None)
            return importlib.import_module("tts")

    def test_synthesize_returns_normalized_audio(self):
        tts_module = self._import_service()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "voice.onnx"
            model_path.touch()
            service = tts_module.TextToSpeechService(str(model_path))

            sample_rate, audio = service.synthesize("Hello.")

        self.assertEqual(sample_rate, 22050)
        np.testing.assert_allclose(audio, [0.0, 0.5, -0.5])

    def test_save_voice_sample_writes_wav(self):
        tts_module = self._import_service()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "voice.onnx"
            output_path = Path(directory) / "response.wav"
            model_path.touch()
            service = tts_module.TextToSpeechService(str(model_path))

            service.save_voice_sample("Hello.", str(output_path))

            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 22050)
                self.assertEqual(wav_file.getnframes(), 3)

    def test_missing_voice_model_has_download_guidance(self):
        tts_module = self._import_service()
        with self.assertRaisesRegex(FileNotFoundError, "piper.download_voices"):
            tts_module.TextToSpeechService("missing-voice.onnx")


if __name__ == "__main__":
    unittest.main()

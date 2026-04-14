import whisper
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_MODEL_CACHE = {}
DEFAULT_WAV = Path(__file__).resolve().parent / "audio.wav"


def _load_model(model_size="base"):
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def _resample_audio(audio, src_rate, tgt_rate=16000):
    if src_rate == tgt_rate:
        return audio
    src_len = len(audio)
    tgt_len = int(src_len * float(tgt_rate) / float(src_rate))
    if tgt_len <= 1:
        return audio
    src_idx = np.linspace(0, src_len - 1, num=src_len)
    tgt_idx = np.linspace(0, src_len - 1, num=tgt_len)
    return np.interp(tgt_idx, src_idx, audio).astype(np.float32)


def _read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = _resample_audio(audio, sr, 16000)
    return audio


def speech_to_text(audio_path=DEFAULT_WAV, model_size="base", language="en"):
    wav_path = Path(audio_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    model = _load_model(model_size=model_size)
    use_fp16 = bool(torch and torch.cuda.is_available())
    audio = _read_audio(wav_path)
    result = model.transcribe(audio, language=language, fp16=use_fp16)
    return result.get("text", "").strip()

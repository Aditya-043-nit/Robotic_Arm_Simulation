import sounddevice as sd
import soundfile as sf
from pathlib import Path

DEFAULT_WAV = Path(__file__).resolve().parent / "audio.wav"


def record_audio(
    filename=DEFAULT_WAV,
    duration=5.0,
    sample_rate=16000,
    channels=1,
    dtype="float32",
):
    wav_path = Path(filename)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording for {duration:.1f}s at {sample_rate}Hz...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
    )
    sd.wait()
    sf.write(str(wav_path), audio, sample_rate)
    print("Saved:", wav_path)
    return str(wav_path)

import argparse
import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

if __package__:
    from .inference.parser import parse_command
    from .stt.audio_recording import record_audio
    from .stt.speech_to_text import speech_to_text
else:
    from inference.parser import parse_command
    from stt.audio_recording import record_audio
    from stt.speech_to_text import speech_to_text

DEFAULT_AUDIO_FILE = Path(__file__).resolve().parent / "stt" / "audio.wav"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech to simple robot command JSON.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Skip audio capture and parse this command text directly.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=DEFAULT_AUDIO_FILE,
        help="Where the audio file is stored/read. Defaults to NLP/stt/audio.wav",
    )
    parser.add_argument(
        "--skip-record",
        action="store_true",
        help="Skip microphone recording and transcribe existing --audio-path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size: tiny, base, small, medium, large.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint for Whisper transcription.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file path to save parsed output.",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.text:
        text = args.text.strip()
    else:
        wav_path = Path(args.audio_path)
        if not args.skip_record:
            record_audio(
                filename=wav_path,
                duration=args.duration,
                sample_rate=args.sample_rate,
                channels=1,
            )
        text = speech_to_text(
            audio_path=wav_path,
            model_size=args.whisper_model,
            language=args.language,
        )

    return parse_command(text)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    result = run_pipeline(args)
    output = json.dumps(result, indent=2, ensure_ascii=True)
    print(output)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

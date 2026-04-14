# NLP Module (Simple Guide)

This module takes speech or text and returns a small JSON with:
- object
- color
- size
- other attributes
- source
- destination

It uses:
- Whisper for speech-to-text
- DistilBERT for intent + slot understanding

## What each file does

- `main.py`: main entry point. Runs record -> transcribe -> parse.
- `stt/audio_recording.py`: records audio and saves it to `NLP/stt/audio.wav` by default.
- `stt/speech_to_text.py`: reads audio (default `NLP/stt/audio.wav`) and transcribes with Whisper.
- `inference/intent_predict.py`: DistilBERT intent model loader/predictor.
- `inference/slot_extractor.py`: DistilBERT slot model loader/extractor.
- `inference/parser.py`: builds final simple JSON output.
- `training/train_intent.py`: trains intent model from `data/commands.csv`.
- `training/train_slot_tagger.py`: trains slot model from `data/slot_commands.jsonl`.
- `data/commands.csv`: intent dataset (1200 samples + header).
- `data/slot_commands.jsonl`: slot dataset (1200 samples).

## Setup

From project root:

```
pip install -r NLP/requirements.txt
```

## Train models

```
python NLP/training/train_intent.py
python NLP/training/train_slot_tagger.py
```

Models are saved in:
- `NLP/models/intent_model`
- `NLP/models/slot_model`

## Run

Text input:

```
python NLP/main.py --text "pick the small red cup from left shelf and place it on center table"
```

Record from mic (audio saved in `NLP/stt/audio.wav`):

```
python NLP/main.py --duration 6
```

Use existing audio file:

```
python NLP/main.py --skip-record --audio-path NLP/stt/audio.wav
```

## JSON output format

```json
{
  "object": "cup",
  "color": "red",
  "size": "small",
  "other_attributes": ["glass", "small", "red"],
  "source": "left shelf",
  "destination": "center table"
}
```

import json
import os
from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "intent_model"
LABEL_MAP_PATH = MODEL_PATH / "label_map.json"


def _heuristic_intent(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ["pick", "grab", "bring", "fetch"]) and any(
        word in text_lower for word in ["place", "put", "drop", "move"]
    ):
        return {"label": "pick_and_place", "confidence": 0.45}
    if any(word in text_lower for word in ["place", "put", "drop"]):
        return {"label": "place_object", "confidence": 0.4}
    if any(word in text_lower for word in ["find", "search", "locate"]):
        return {"label": "search_object", "confidence": 0.4}
    if any(word in text_lower for word in ["pick", "grab", "bring", "fetch"]):
        return {"label": "pick_object", "confidence": 0.4}
    return {"label": "unknown", "confidence": 0.2}


class IntentModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.inv_label_map = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = False
        self.error = None

    def load(self):
        if self.ready:
            return

        if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
            self.error = (
                f"Intent model not found at {MODEL_PATH}. "
                "Run NLP/training/train_intent.py first."
            )
            self.ready = True
            return

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
            self.model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            self.inv_label_map = {int(v): k for k, v in label_map.items()}
            self.model.to(self.device)
        except Exception as exc:  # pragma: no cover
            self.error = f"Failed to load intent model: {exc}"
        finally:
            self.ready = True

    def predict(self, text):
        self.load()
        if self.model is None or self.tokenizer is None or self.inv_label_map is None:
            return _heuristic_intent(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidence, pred = torch.max(probabilities, dim=-1)

        pred_idx = int(pred.item())
        return {
            "label": self.inv_label_map.get(pred_idx, "unknown"),
            "confidence": float(confidence.item()),
        }


_INTENT_MODEL = IntentModel()


def predict_intent(text):
    return _INTENT_MODEL.predict(text)

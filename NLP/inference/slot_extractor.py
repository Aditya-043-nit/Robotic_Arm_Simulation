import json
from pathlib import Path

import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "slot_model"
LABEL_MAP_PATH = MODEL_PATH / "label_map.json"

FALLBACK_OBJECTS = {
    "cup",
    "bottle",
    "box",
    "apple",
    "banana",
    "block",
    "can",
    "tool",
    "tray",
    "screwdriver",
    "remote",
    "mug",
}
FALLBACK_COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "gray",
    "brown",
}
FALLBACK_SIZES = {"small", "medium", "large", "tiny", "big"}


def _dedupe_keep_order(items):
    seen = set()
    output = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        output.append(normalized)
        seen.add(normalized)
    return output


def _empty_result():
    return {
        "object": {"name": None, "color": None, "size": None, "attributes": []},
        "source": {"description": None},
        "destination": {"description": None},
        "entities": [],
    }


def _heuristic_slots(text):
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    result = _empty_result()

    for token in tokens:
        if token in FALLBACK_OBJECTS and not result["object"]["name"]:
            result["object"]["name"] = token
        if token in FALLBACK_COLORS and not result["object"]["color"]:
            result["object"]["color"] = token
        if token in FALLBACK_SIZES and not result["object"]["size"]:
            result["object"]["size"] = token

    source_markers = [" from ", " off ", " out of "]
    destination_markers = [" to ", " into ", " onto ", " on ", " in "]

    text_padded = f" {text.lower()} "

    for marker in source_markers:
        if marker in text_padded:
            src_part = text_padded.split(marker, 1)[1]
            for cut in [" and ", " then ", " place ", " put ", " drop ", " move "]:
                if cut in src_part:
                    src_part = src_part.split(cut, 1)[0]
                    break
            result["source"]["description"] = src_part.strip()
            break

    for marker in destination_markers:
        if marker in text_padded:
            dst_part = text_padded.rsplit(marker, 1)[1]
            for cut in [" and ", " then "]:
                if cut in dst_part:
                    dst_part = dst_part.split(cut, 1)[0]
                    break
            result["destination"]["description"] = dst_part.strip()
            break

    return result


class SlotModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.id_to_label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = False

    def load(self):
        if self.ready:
            return

        if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
            self.ready = True
            return

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
            self.model = DistilBertForTokenClassification.from_pretrained(str(MODEL_PATH))
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            self.id_to_label = {int(v): k for k, v in label_map.items()}
            self.model.to(self.device)
        except Exception:  # pragma: no cover
            self.model = None
            self.tokenizer = None
            self.id_to_label = None
        finally:
            self.ready = True

    def _predict_word_labels(self, text):
        words = text.strip().split()
        if not words:
            return [], []

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()

        per_word_labels = []
        per_word_conf = []
        previous_word = None

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word:
                continue
            label = self.id_to_label.get(pred_ids[token_idx], "O")
            confidence = float(probs[token_idx, pred_ids[token_idx]].item())
            per_word_labels.append(label)
            per_word_conf.append(confidence)
            previous_word = word_idx

        return words[: len(per_word_labels)], list(zip(per_word_labels, per_word_conf))

    @staticmethod
    def _merge_entities(words, labels_with_confidence):
        entities = []
        current = None

        for idx, (word, (label, score)) in enumerate(zip(words, labels_with_confidence)):
            if label == "O":
                if current:
                    current["confidence"] = current["confidence"] / max(current["count"], 1)
                    current.pop("count", None)
                    entities.append(current)
                    current = None
                continue

            prefix, slot_type = label.split("-", 1) if "-" in label else ("B", label)

            if prefix == "B" or not current or current["slot"] != slot_type:
                if current:
                    current["confidence"] = current["confidence"] / max(current["count"], 1)
                    current.pop("count", None)
                    entities.append(current)
                current = {
                    "slot": slot_type,
                    "text": word,
                    "confidence": score,
                    "count": 1,
                    "start_word": idx,
                    "end_word": idx,
                }
            else:
                current["text"] += f" {word}"
                current["confidence"] += score
                current["count"] += 1
                current["end_word"] = idx

        if current:
            current["confidence"] = current["confidence"] / max(current["count"], 1)
            current.pop("count", None)
            entities.append(current)

        return entities

    def extract(self, text):
        self.load()
        if not self.model or not self.tokenizer or not self.id_to_label:
            return _heuristic_slots(text)

        words, labels_with_confidence = self._predict_word_labels(text)
        entities = self._merge_entities(words, labels_with_confidence)

        result = _empty_result()
        result["entities"] = entities

        object_attrs = []

        for entity in entities:
            slot = entity["slot"]
            slot_text = entity["text"].strip()
            if slot == "OBJ_NAME" and not result["object"]["name"]:
                result["object"]["name"] = slot_text
            elif slot == "OBJ_COLOR" and not result["object"]["color"]:
                result["object"]["color"] = slot_text
            elif slot == "OBJ_SIZE" and not result["object"]["size"]:
                result["object"]["size"] = slot_text
            elif slot == "OBJ_ATTR":
                object_attrs.append(slot_text)
            elif slot == "SRC_LOC" and not result["source"]["description"]:
                result["source"]["description"] = slot_text
            elif slot == "DST_LOC" and not result["destination"]["description"]:
                result["destination"]["description"] = slot_text

        if result["object"]["size"]:
            object_attrs.append(result["object"]["size"])
        if result["object"]["color"]:
            object_attrs.append(result["object"]["color"])

        result["object"]["attributes"] = _dedupe_keep_order(object_attrs)
        return result


_SLOT_MODEL = SlotModel()


def extract_slots(text):
    return _SLOT_MODEL.extract(text)

from .intent_predict import predict_intent
from .slot_extractor import extract_slots


def parse_command(text):
    clean_text = " ".join((text or "").strip().split())
    _ = predict_intent(clean_text)
    slots = extract_slots(clean_text)
    obj = slots.get("object", {})
    src = slots.get("source", {})
    dst = slots.get("destination", {})

    return {
        "object": obj.get("name"),
        "color": obj.get("color"),
        "size": obj.get("size"),
        "other_attributes": obj.get("attributes", []),
        "source": src.get("description"),
        "destination": dst.get("description"),
    }

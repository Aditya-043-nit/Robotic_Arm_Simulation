# coordinate_builder.py

def build_coordinates(task: dict, detections: list[dict]) -> dict:
    """
    Match detections to pickup object and drop zone from the parsed task.
    Returns final JSON output for the master controller.

    task       : output from nlp_parser.parse_order()
    detections : output from detector.detect_objects()
    """

    obj_label  = task["object"].lower()
    drop_label = task["drop_zone"].lower()

    pickup_detection = None
    drop_detection   = None

    for d in detections:
        detected = d["label"].lower()

        if pickup_detection is None and (
            obj_label in detected or detected in obj_label
        ):
            pickup_detection = d

        if drop_detection is None and (
            drop_label in detected or detected in drop_label
        ):
            drop_detection = d

    # Build result
    result = {
        "status": "success",
        "order" : task,
        "pickup": None,
        "drop"  : None,
        "warnings": []
    }

    if pickup_detection:
        cx, cy = pickup_detection["center"]
        result["pickup"] = {
            "label"     : pickup_detection["label"],
            "confidence": pickup_detection["confidence"],
            "x_px"      : cx,
            "y_px"      : cy,
            "bbox"      : pickup_detection["bbox"]
        }
    else:
        result["status"] = "partial"
        result["warnings"].append(f"Object '{task['object']}' not detected in frame")

    if drop_detection:
        cx, cy = drop_detection["center"]
        result["drop"] = {
            "label"     : drop_detection["label"],
            "confidence": drop_detection["confidence"],
            "x_px"      : cx,
            "y_px"      : cy,
            "bbox"      : drop_detection["bbox"]
        }
    else:
        result["status"] = "partial"
        result["warnings"].append(f"Drop zone '{task['drop_zone']}' not detected in frame")

    if not pickup_detection and not drop_detection:
        result["status"] = "failed"

    return result
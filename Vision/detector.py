# detector.py
import cv2
import json
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_MODEL, DETECT_CONF

model = YOLO(YOLO_MODEL)

WARMUP_FRAMES   = 10
RESULTS_DIR     = Path("data/captures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE    = RESULTS_DIR / "session_results.json"   # single file, all captures


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_session() -> dict:
    """Load existing session file, or create a fresh one."""
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    return {
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "captures"  : []
    }


def _save_session(session: dict) -> None:
    SESSION_FILE.write_text(
        json.dumps(session, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _build_capture_record(
    task        : dict,
    detections  : list[dict],
    capture_num : int
) -> dict:
    """
    Build a single capture record from task + detections.
    Extracts pickup (object) and drop (drop_zone) coordinates specifically.
    """
    obj_label  = task.get("object",    "").lower()
    drop_label = task.get("drop_zone", "").lower()

    pickup_coord = None
    drop_coord   = None

    for d in detections:
        detected = d["label"].lower()

        if pickup_coord is None and (
            obj_label in detected or detected in obj_label
        ):
            cx, cy = d["center"]
            pickup_coord = {
                "label"     : d["label"],
                "confidence": d["confidence"],
                "x_px"      : cx,
                "y_px"      : cy,
                "bbox"      : d["bbox"]
            }

        if drop_coord is None and (
            drop_label in detected or detected in drop_label
        ):
            cx, cy = d["center"]
            drop_coord = {
                "label"     : d["label"],
                "confidence": d["confidence"],
                "x_px"      : cx,
                "y_px"      : cy,
                "bbox"      : d["bbox"]
            }

    warnings = []
    if not pickup_coord:
        warnings.append(f"Object '{task.get('object')}' not found in frame")
    if not drop_coord:
        warnings.append(f"Drop zone '{task.get('drop_zone')}' not found in frame")

    status = "success" if (pickup_coord and drop_coord) else \
             "partial" if (pickup_coord or drop_coord) else "failed"

    return {
        "capture_num"      : capture_num,
        "captured_at"      : datetime.now(timezone.utc).isoformat(),
        "status"           : status,
        "order"            : task,
        "pickup"           : pickup_coord,
        "drop"             : drop_coord,
        "all_detections"   : detections,
        "warnings"         : warnings
    }


def _detect_objects(frame: np.ndarray, target_labels: list[str]) -> list[dict]:
    """Run YOLO. Return detections matching target_labels."""
    results = model(frame, conf=DETECT_CONF, verbose=False)[0]

    detections = []
    for box in results.boxes:
        label = results.names[int(box.cls)]

        matched = any(
            tgt.lower() in label.lower() or label.lower() in tgt.lower()
            for tgt in target_labels
        )
        if not matched:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "label"     : label,
            "confidence": round(float(box.conf), 3),
            "bbox"      : [x1, y1, x2, y2],
            "center"    : [(x1 + x2) // 2, (y1 + y2) // 2]
        })

    return detections


def _draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes + centers on frame."""
    vis = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cx, cy          = d["center"]
        label           = f"{d['label']} {d['confidence']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis


# ── Main public function ──────────────────────────────────────────────────────

def run_live_detection(task: dict) -> dict:
    """
    Opens live webcam feed. Detects objects continuously.
    SPACE  → capture current frame, extract coordinates, append to session JSON.
    Q      → quit and return the full session.

    Args:
        task : parsed task from nlp_parser.parse_order()
                {action, object, pickup_zone, drop_zone}

    Returns:
        Full session dict (also saved to data/captures/session_results.json)
    """
    target_labels = [task.get("object", ""), task.get("drop_zone", "")]
    target_labels = [t for t in target_labels if t]   # remove empty strings

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAMERA_INDEX}. "
            "Check CAMERA_INDEX in config.py"
        )

    # Warmup
    print(f"  Warming up camera ({WARMUP_FRAMES} frames)...", end=" ", flush=True)
    for _ in range(WARMUP_FRAMES):
        cap.read()
    print("ready.")
    print(f"  Looking for: {target_labels}")
    print("  SPACE = capture frame   |   Q = quit\n")

    session     = _load_session()
    capture_num = len(session["captures"]) + 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Flip so it acts like a normal mirror camera
            frame = cv2.flip(frame, 1)

            # Always detect on live feed
            detections = _detect_objects(frame, target_labels)

            # Draw live
            preview = _draw_detections(frame, detections)

            # HUD
            cv2.putText(preview,
                        f"Looking for: {', '.join(target_labels)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(preview,
                        f"Detected: {len(detections)}  |  Captures: {capture_num - 1}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(preview,
                        "SPACE = capture   Q = quit",
                        (10, CAMERA_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Robot Vision", preview)

            key = cv2.waitKey(1) & 0xFF

            # ── SPACE: capture this frame ─────────────────────────────────
            if key == ord(" "):
                print(f"  [capture {capture_num}] Processing frame...", end=" ", flush=True)

                record = _build_capture_record(task, detections, capture_num)
                session["captures"].append(record)
                _save_session(session)

                # Flash green border to confirm capture
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0),
                              (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1),
                              (0, 255, 0), 8)
                cv2.imshow("Robot Vision", flash)
                cv2.waitKey(200)   # show flash for 200ms

                print(f"saved. status={record['status']}")
                if record["pickup"]:
                    print(f"    pickup → {record['pickup']['label']} "
                          f"at ({record['pickup']['x_px']}, {record['pickup']['y_px']})")
                if record["drop"]:
                    print(f"    drop   → {record['drop']['label']} "
                          f"at ({record['drop']['x_px']}, {record['drop']['y_px']})")
                if record["warnings"]:
                    for w in record["warnings"]:
                        print(f"    warning: {w}")
                print()

                capture_num += 1

            # ── Q: quit ───────────────────────────────────────────────────
            elif key == ord("q"):
                print(f"\n  Camera stopped. Total captures this run: {capture_num - 1}")
                print(f"  Saved to: {SESSION_FILE}")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return session


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a parsed task (normally comes from nlp_parser.parse_order())
    mock_task = {
        "action"     : "put",
        "object"     : "banana",     # change to what's in front of your camera
        # "pickup_zone": "table",
        "drop_zone"  : "notebook"         # change to what's in front of your camera
    }

    print("Robot Vision — Live Detection")
    print(f"Task: {mock_task}\n")

    session = run_live_detection(mock_task)

    print(f"\nSession summary: {len(session['captures'])} capture(s) saved")
    print(f"File: {SESSION_FILE.resolve()}")
"""Live person detection on drone (or any) video.

Run:
    python figure_recognition/detect_live.py

Set SOURCE below to:
  - a local video path  (e.g. "samples/drone_test.mp4")
  - an RTSP URL         (e.g. "rtsp://<drone-bridge-ip>:8554/live")
  - 0                   (Mac built-in webcam)

Outputs:
  - cv2 window: video + bbox overlay + right sidebar with last 18 detection log lines
  - results/detections.jsonl: one JSON line per second (tail -f for live)
"""

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===== CONFIG (edit before running) ============================================
HERE       = Path(__file__).resolve().parent
SOURCE     = HERE / "samples" / "drone_test.mp4"   # swap for RTSP URL or 0
MODEL_NAME = "yolov8m.pt"                          # auto-downloads to models/ on first run
MODEL_DIR  = HERE / "models"
LOG_PATH   = HERE / "results" / "detections.jsonl"

CONF_THR        = 0.5
DEVICE          = "mps"   # "mps" on Mac arm64, "cuda" on linux GPU, "cpu" anywhere
LOG_INTERVAL_S  = 1.0
WINDOW_NAME     = "Drone Person Detection"
SIDEBAR_W       = 360
SIDEBAR_LINES   = 18
RECONNECT_WAIT  = 1.0
# ==============================================================================


def open_capture(source):
    cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort: try to keep latency low
    return cap


def utc_now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def build_sidebar(h, recent_log_lines):
    sb = np.full((h, SIDEBAR_W, 3), 30, dtype=np.uint8)
    cv2.putText(sb, "DETECTIONS (1 Hz)", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 230, 240), 1, cv2.LINE_AA)
    cv2.line(sb, (8, 38), (SIDEBAR_W - 8, 38), (80, 80, 80), 1)
    for i, line in enumerate(reversed(recent_log_lines)):
        y = 64 + i * 19
        if y > h - 8:
            break
        color = (130, 220, 130) if "person" in line else (160, 160, 160)
        cv2.putText(sb, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return sb


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_NAME
    if model_path.exists():
        print(f"loading YOLO from {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"downloading YOLO {MODEL_NAME} (first run) ...")
        model = YOLO(MODEL_NAME)
        # ultralytics drops the file into cwd; move it to models/ for next run
        cwd_pt = Path.cwd() / MODEL_NAME
        if cwd_pt.exists():
            cwd_pt.rename(model_path)
            print(f"cached weights to {model_path}")

    print(f"source:  {SOURCE}")
    print(f"device:  {DEVICE}")
    print(f"log:     {LOG_PATH}  (tail -f for live)")
    print("press ESC in the video window to quit")

    cap = open_capture(SOURCE)
    log_file = open(LOG_PATH, "a", buffering=1)
    recent_log_lines = deque(maxlen=SIDEBAR_LINES)

    frame_idx     = 0
    last_log_ts   = 0.0
    t_prev        = time.time()
    fps_smooth    = 0.0

    try:
        while True:
            # drain any backed-up RTSP frames; keep only the freshest
            for _ in range(4):
                cap.grab()
            ok, frame = cap.retrieve() if cap.grab() else (False, None)

            if not ok or frame is None:
                print(f"stream read failed; reconnecting in {RECONNECT_WAIT}s")
                cap.release()
                time.sleep(RECONNECT_WAIT)
                cap = open_capture(SOURCE)
                continue

            frame_idx += 1

            # YOLO inference, person class only
            results = model.predict(
                frame, classes=[0], conf=CONF_THR,
                device=DEVICE, verbose=False,
            )
            r = results[0]

            # pick highest-conf person (single-person assumption)
            detection = None
            if len(r.boxes) > 0:
                confs = r.boxes.conf.cpu().numpy()
                idx   = int(confs.argmax())
                box   = r.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
                detection = {"bbox": box, "confidence": float(confs[idx])}

            # draw bbox + label
            if detection:
                x1, y1, x2, y2 = detection["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {detection['confidence']:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)

            # fps counter
            now = time.time()
            dt  = now - t_prev
            t_prev = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt
            cv2.putText(frame, f"{fps_smooth:5.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            # 1 Hz log
            if now - last_log_ts >= LOG_INTERVAL_S:
                last_log_ts = now
                ts = utc_now_iso()
                if detection:
                    entry = {
                        "ts": ts, "frame": frame_idx, "person": True,
                        "confidence": round(detection["confidence"], 3),
                        "bbox": detection["bbox"],
                    }
                    log_line = f"{ts[11:19]}  person {detection['confidence']:.2f}"
                else:
                    entry = {"ts": ts, "frame": frame_idx, "person": False,
                             "confidence": None, "bbox": None}
                    log_line = f"{ts[11:19]}  ----"
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                recent_log_lines.append(log_line)

            # composite video + sidebar and show
            sidebar   = build_sidebar(frame.shape[0], recent_log_lines)
            composite = np.hstack([frame, sidebar])
            cv2.imshow(WINDOW_NAME, composite)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cap.release()
        log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

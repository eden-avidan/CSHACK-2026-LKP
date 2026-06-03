"""Live person detection on drone (or any) video, served as a browser page.

Run:
    python figure_recognition/detect_live.py

Then open:
    http://localhost:8000/

Layout:
    [ annotated video stream ]    [ detection log, 1 Hz, auto-scrolling ]

SOURCE (set below) can be:
  - a local video path  (e.g. "samples/drone_test.mp4")
  - an RTSP URL         (e.g. "rtsp://<drone-bridge-ip>:8554/live")
  - 0                   (Mac built-in webcam)

JSONL log mirrored to figure_recognition/results/detections.jsonl (tail -f for terminal view).
"""

import json
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# ===== CONFIG (edit before running) ============================================
HERE        = Path(__file__).resolve().parent
SOURCE      = 0                              # 0 = webcam; or "rtsp://..."; or a video path
MODEL_NAME  = "yolov8m.pt"                   # auto-downloads on first run
MODEL_DIR   = HERE / "models"
LOG_PATH    = HERE / "results" / "detections.jsonl"

CONF_THR        = 0.5
DEVICE          = "mps"                      # "mps" (Mac), "cuda" (Linux GPU), or "cpu"
LOG_INTERVAL_S  = 1.0
RECONNECT_WAIT  = 1.0
JPEG_QUALITY    = 85
WEBCAM_REQ_W    = 1280                       # only applied when SOURCE is int (webcam)
WEBCAM_REQ_H    = 720
HOST            = "0.0.0.0"
PORT            = 8000
# ==============================================================================


app = Flask(__name__)

# shared state (worker thread -> http handlers)
_latest_jpeg   = b""
_jpeg_lock     = threading.Lock()
_log_subs      = []                          # list of queue.Queue[str], one per SSE client
_log_subs_lock = threading.Lock()
_log_recent    = deque(maxlen=200)           # served once on page load for backfill


def open_capture(source):
    is_int = isinstance(source, int)
    cap = cv2.VideoCapture(source if is_int else str(source))
    if is_int:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_REQ_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_REQ_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def utc_now_iso():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def publish_log(entry_dict):
    line = json.dumps(entry_dict)
    _log_recent.append(line)
    with _log_subs_lock:
        for q in list(_log_subs):
            try:
                q.put_nowait(line)
            except queue.Full:
                pass


def worker():
    global _latest_jpeg

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_NAME
    if model_path.exists():
        print(f"[worker] loading YOLO from {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"[worker] downloading YOLO {MODEL_NAME} (first run) ...")
        model = YOLO(MODEL_NAME)
        cwd_pt = Path.cwd() / MODEL_NAME
        if cwd_pt.exists():
            cwd_pt.rename(model_path)
            print(f"[worker] cached weights to {model_path}")

    print(f"[worker] source: {SOURCE}  device: {DEVICE}")
    cap = open_capture(SOURCE)
    log_file = open(LOG_PATH, "a", buffering=1)

    frame_idx   = 0
    last_log_ts = 0.0
    t_prev      = time.time()
    fps_smooth  = 0.0

    while True:
        for _ in range(4):
            cap.grab()
        ok, frame = cap.retrieve() if cap.grab() else (False, None)
        if not ok or frame is None:
            print(f"[worker] stream read failed; reconnecting in {RECONNECT_WAIT}s")
            cap.release()
            time.sleep(RECONNECT_WAIT)
            cap = open_capture(SOURCE)
            continue

        frame_idx += 1

        results = model.predict(
            frame, classes=[0], conf=CONF_THR,
            device=DEVICE, verbose=False,
        )
        r = results[0]

        detection = None
        if len(r.boxes) > 0:
            confs = r.boxes.conf.cpu().numpy()
            idx   = int(confs.argmax())
            box   = r.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
            detection = {"bbox": box, "confidence": float(confs[idx])}

        if detection:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {detection['confidence']:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # fps overlay
        now = time.time()
        dt  = now - t_prev
        t_prev = now
        if dt > 0:
            instant = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant if fps_smooth else instant
        cv2.putText(frame, f"{fps_smooth:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        # encode + publish jpeg
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok_enc:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

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
            else:
                entry = {"ts": ts, "frame": frame_idx, "person": False,
                         "confidence": None, "bbox": None}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            publish_log(entry)


# ===== HTTP =====================================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Drone Person Detection</title>
  <style>
    :root { color-scheme: dark; }
    html, body { margin: 0; padding: 0; height: 100%; background: #0b0b0d; color: #d8d8d8;
                 font-family: -apple-system, BlinkMacSystemFont, "SF Mono", Menlo, monospace; }
    .row { display: flex; height: 100vh; }
    .video { flex: 3; display: flex; align-items: center; justify-content: center;
             background: #000; min-width: 0; }
    .video img { max-width: 100%; max-height: 100%; display: block; }
    .log { flex: 1; min-width: 320px; max-width: 480px; padding: 16px 18px;
           border-left: 1px solid #222; display: flex; flex-direction: column;
           overflow: hidden; }
    .log h2 { margin: 0 0 12px 0; font-size: 13px; letter-spacing: 0.08em;
              color: #8aa; text-transform: uppercase; font-weight: 600; }
    .log .stream { flex: 1; overflow-y: auto; font-size: 13px; line-height: 1.7;
                   white-space: nowrap; }
    .entry { padding: 2px 0; }
    .entry .t  { color: #667; margin-right: 10px; }
    .entry.person .label { color: #6df16d; }
    .entry.empty  .label { color: #555; }
    .entry .conf { color: #d8d8d8; margin-left: 8px; }
    .topbar { position: fixed; top: 8px; left: 8px; font-size: 11px;
              background: rgba(0,0,0,0.4); padding: 4px 8px; border-radius: 4px;
              color: #9ab; }
  </style>
</head>
<body>
  <div class="topbar">F11 to fullscreen &nbsp;·&nbsp; Ctrl-C in terminal to quit</div>
  <div class="row">
    <div class="video"><img src="/video" alt="live"></div>
    <div class="log">
      <h2>Detections (1 Hz)</h2>
      <div class="stream" id="stream"></div>
    </div>
  </div>
<script>
  const stream = document.getElementById("stream");
  function addEntry(o) {
    const div = document.createElement("div");
    div.className = "entry " + (o.person ? "person" : "empty");
    const t = (o.ts || "").slice(11, 19);
    if (o.person) {
      div.innerHTML = `<span class="t">${t}</span><span class="label">person</span><span class="conf">${o.confidence.toFixed(2)}</span>`;
    } else {
      div.innerHTML = `<span class="t">${t}</span><span class="label">----</span>`;
    }
    stream.prepend(div);
    while (stream.childElementCount > 200) stream.removeChild(stream.lastChild);
  }
  const es = new EventSource("/log");
  es.onmessage = ev => { try { addEntry(JSON.parse(ev.data)); } catch (e) {} };
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/video")
def video():
    boundary = b"--frame"
    def gen():
        last_sent = b""
        while True:
            with _jpeg_lock:
                buf = _latest_jpeg
            if buf and buf is not last_sent:
                last_sent = buf
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n")
            time.sleep(1.0 / 30)   # cap stream at ~30 fps
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/log")
def log_stream():
    def gen():
        q = queue.Queue(maxsize=1000)
        with _log_subs_lock:
            _log_subs.append(q)
        try:
            # backfill recent entries so a late-joiner sees context
            for line in list(_log_recent):
                yield f"data: {line}\n\n"
            while True:
                line = q.get()
                yield f"data: {line}\n\n"
        finally:
            with _log_subs_lock:
                if q in _log_subs:
                    _log_subs.remove(q)
    return Response(gen(), mimetype="text/event-stream")


def main():
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    print(f"[http] serving on http://{HOST}:{PORT}/  (open in your browser, F11 to fullscreen)")
    # use threaded=True so MJPEG + SSE can coexist with future HTTP hits
    app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

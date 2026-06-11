"""Offline person detection on a video file → JSONL output. No HTTP, no browser.

Run:
    python figure_recognition/detect_offline.py
    python figure_recognition/detect_offline.py path/to/clip.mp4
    python figure_recognition/detect_offline.py path/to/clip.mp4 --every-n 5 --imgsz 960

Output (one JSON object per processed frame, newline-delimited):
    {"ts": "00:00:00.500", "frame": 15, "person": true,  "confidence": 0.967, "bbox": [360, 12, 531, 366]}
    {"ts": "00:00:01.000", "frame": 30, "person": false, "confidence": null,  "bbox": null}

`ts` is the video timestamp (relative to clip start), not wall-clock.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ===== CONFIG ==================================================================
HERE        = Path(__file__).resolve().parent
SOURCE      = HERE / "samples" / "drone_test.mp4"
MODEL_NAME  = "yolo26x.pt"
MODEL_DIR   = HERE / "models"

CONF_THR    = 0.5
DEVICE      = "mps"          # "mps" (Mac), "cuda" (Linux GPU), or "cpu"
INFER_SIZE  = 1280
EVERY_N     = 1              # 1 = every frame, 5 = every 5th frame
# ==============================================================================


def fmt_ts(seconds: float) -> str:
    """Format video-time seconds as HH:MM:SS.mmm."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def annotate(frame, bbox, confidence):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"person {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)


def detect(video_path: Path, out_path: Path, render_path: Path | None,
           model_name: str, device: str,
           imgsz: int, conf_thr: float, every_n: int) -> None:
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        print(f"[load] {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"[load] downloading {model_name} ...")
        model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[err] cannot open {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {video_path.name}  {width}x{height}  fps={fps:.2f}  frames={total}  every_n={every_n}")
    print(f"[model] device={device}  imgsz={imgsz}  conf={conf_thr}")
    print(f"[out]   {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(out_path, "w", buffering=1)

    writer = None
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(render_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            sys.exit(f"[err] cannot open writer for {render_path}")
        print(f"[render] {render_path}")

    found = 0
    processed = 0
    half = device == "cuda"
    t0 = time.time()
    last_detection = None   # carry-forward annotation between inferenced frames

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        run_inference = (frame_idx % every_n == 0)

        if run_inference:
            results = model.predict(
                frame, classes=[0], conf=conf_thr,
                device=device, imgsz=imgsz, half=half, verbose=False,
            )
            boxes = results[0].boxes

            if len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                i = int(confs.argmax())
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                last_detection = (bbox, float(confs[i]))
                entry = {
                    "ts": fmt_ts(frame_idx / fps),
                    "frame": frame_idx,
                    "person": True,
                    "confidence": round(float(confs[i]), 3),
                    "bbox": bbox,
                }
                found += 1
            else:
                last_detection = None
                entry = {
                    "ts": fmt_ts(frame_idx / fps),
                    "frame": frame_idx,
                    "person": False,
                    "confidence": None,
                    "bbox": None,
                }

            log.write(json.dumps(entry) + "\n")
            processed += 1

            if processed % 30 == 0:
                elapsed = time.time() - t0
                pct = 100.0 * frame_idx / total if total else 0.0
                print(f"  [{pct:5.1f}%] frame {frame_idx}/{total}  "
                      f"found={found}/{processed}  "
                      f"speed={processed/elapsed:.1f} fps")

        if writer is not None:
            if last_detection is not None:
                annotate(frame, last_detection[0], last_detection[1])
            writer.write(frame)

    cap.release()
    log.close()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    print()
    print(f"[done] inferenced {processed} frames in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} fps)")
    print(f"[done] person found in {found}/{processed} frames "
          f"({100*found/max(processed,1):.1f}%)")
    print(f"[done] jsonl:  {out_path}")
    if render_path is not None:
        print(f"[done] video:  {render_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(SOURCE),
                    help="path to input video (default: bundled drone_test.mp4)")
    ap.add_argument("--out",        default=None, help="output .jsonl (default: results/<video>.jsonl)")
    ap.add_argument("--render-out", default=None, help="annotated video output (default: results/<video>_annotated.mp4)")
    ap.add_argument("--no-render",  action="store_true", help="skip writing the annotated video")
    ap.add_argument("--model",      default=MODEL_NAME)
    ap.add_argument("--device",     default=DEVICE, choices=["mps", "cuda", "cpu"])
    ap.add_argument("--imgsz",      type=int, default=INFER_SIZE)
    ap.add_argument("--conf",       type=float, default=CONF_THR)
    ap.add_argument("--every-n",    type=int, default=EVERY_N,
                    help="run inference on every Nth frame (1=all frames). "
                         "Skipped frames carry forward the last bbox in the rendered video.")
    args = ap.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"[err] video not found: {video_path}")

    out_path = Path(args.out) if args.out else HERE / "results" / f"{video_path.stem}.jsonl"

    render_path = None
    if not args.no_render:
        render_path = Path(args.render_out) if args.render_out \
            else HERE / "results" / f"{video_path.stem}_annotated.mp4"

    detect(video_path, out_path, render_path, args.model, args.device,
           args.imgsz, args.conf, args.every_n)


if __name__ == "__main__":
    main()

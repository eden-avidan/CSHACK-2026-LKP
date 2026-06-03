# CSHACK-2026-LKP

Real-time person detection on a drone (DJI Mini 4 Pro) video feed, with per-frame confidence and a live JSONL log. Runtime detector: **YOLOv8** via Ultralytics; runs on Apple Silicon (MPS), CUDA, or CPU.

## Setup

### 1. Create the conda environment

Mac (Apple Silicon, MPS):
```bash
conda env create -f environment.macos.yaml
conda activate cshack
```

Linux + NVIDIA GPU (e.g. slurm with H100):
```bash
conda env create -f environment.yaml
conda activate cshack
```

Both env files share the same name (`cshack`) and pip dependencies; only the PyTorch backend differs.

### 2. (Optional) OpenMMLab stack — only if you want RTMPose skeleton stretch goal

```bash
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

## Running

```bash
python figure_recognition/detect_live.py
```

Edit the `CONFIG` block at the top of `figure_recognition/detect_live.py` to point `SOURCE` at:
- a local video file path (`Path / "samples/drone_test.mp4"`)
- an RTSP URL (drone bridge), e.g. `"rtsp://192.168.1.10:8554/live"`
- `0` for the Mac built-in webcam

YOLO weights (`yolov8m.pt`, ~50 MB) auto-download on first run into `figure_recognition/models/`.

The script opens a single window with the video, bounding boxes, FPS counter, and a side-panel showing the last ~18 detection log entries. Per-second JSON log lines are written to `figure_recognition/results/detections.jsonl` — `tail -f` it for a live audience-visible stream.

Press `ESC` in the video window to quit.

## Repo layout

```
.
├── README.md
├── environment.yaml              conda env (Linux + CUDA)
├── environment.macos.yaml        conda env (Mac Apple Silicon, MPS)
├── .gitignore                    ignores figure_recognition/{models,results}/ and skills/
└── figure_recognition/
    ├── README.md
    ├── detect_live.py            main script: drone RTSP -> YOLO -> overlay + log
    ├── models/                   model weights (auto-downloaded, not in git)
    └── results/                  detections.jsonl + future recordings (not in git)
```

## Stretch goal — RTMPose skeleton overlay

If you want pose keypoints layered on top of the person bbox, download these into `figure_recognition/models/`:

```bash
mkdir -p figure_recognition/models
curl -L -o figure_recognition/models/rtmpose-l_384x288.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"
curl -L -o figure_recognition/models/rtmdet-m.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
```

(See `skills/mmpose_expert.md` on the dev box for the integration plan.)

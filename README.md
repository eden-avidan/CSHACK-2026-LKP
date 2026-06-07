# CSHACK-2026-LKP

Real-time person detection on a drone (DJI Mini 4 Pro) video feed, with per-frame confidence and a live JSONL log. Runtime detector: **YOLO26** via Ultralytics; runs on Apple Silicon (MPS), CUDA, or CPU.

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

Then open **<http://localhost:8000/>** in your browser. Hit `F11` for fullscreen on demo day.

Edit the `CONFIG` block at the top of `figure_recognition/detect_live.py` to point `SOURCE` at:
- `0` for the Mac built-in webcam
- a local video file path (`HERE / "samples" / "drone_test.mp4"`)
- an RTSP URL (drone bridge), e.g. `"rtsp://192.168.1.10:8554/live"`

YOLO weights (`yolo26m.pt`) auto-download on first run into `figure_recognition/models/`.

The browser page is a flexbox split: annotated video on the left (auto-scales to viewport), live detection log on the right (one entry per second, newest on top, color-coded green when a person is present). Per-second JSON log lines are mirrored to `figure_recognition/results/detections.jsonl` — `tail -f` it for a terminal view.

Press `Ctrl-C` in the terminal to quit.

## Repo layout

```
.
├── README.md
├── environment.yaml              conda env (Linux + CUDA)
├── environment.macos.yaml        conda env (Mac Apple Silicon, MPS)
├── .gitignore                    ignores models/, results/, skills/, .DS_Store
└── figure_recognition/
    ├── README.md
    ├── detect_live.py            main script: drone RTSP -> YOLO -> overlay + log
    ├── samples/
    │   └── drone_test.mp4        bundled test clip (~44 MB)
    ├── models/                   model weights (auto-downloaded, not in git)
    └── results/                  detections.jsonl + future recordings (not in git)
```

## Quick test on the bundled sample

```bash
conda activate cshack
# in figure_recognition/detect_live.py, set SOURCE = HERE / "samples" / "drone_test.mp4"
python figure_recognition/detect_live.py
# then open http://localhost:8000/
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

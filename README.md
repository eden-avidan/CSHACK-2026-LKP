# CSHACK-2026-LKP

Real-time skeleton overlay on video with per-keypoint confidence, built on [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate cshack
```

This installs PyTorch 2.0.1 + CUDA 11.8 + OpenCV + ONNX runtime + `openmim`.

### 2. Install the OpenMMLab stack

```bash
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

### 3. Download model weights (not tracked in git)

Pose model — **RTMPose-l @ 384x288, 78.3 AP on COCO 17-keypoint** (~107 MB):

```bash
mkdir -p figure_recognition
curl -L -o figure_recognition/rtmpose-l_384x288.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth"
```

Person detector — **RTMDet-m** (~95 MB, paired with RTMPose-l):

```bash
curl -L -o figure_recognition/rtmdet-m.pth \
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
```

## Repo layout

```
.
├── README.md
├── environment.yaml          conda env spec (name: cshack)
├── .gitignore                ignores figure_recognition/ and skills/
└── figure_recognition/       model weights (downloaded, not in git)
```

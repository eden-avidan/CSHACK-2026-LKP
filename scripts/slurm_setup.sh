#!/usr/bin/env bash
# Run this ON the slurm gateway (after `ssh eavidan@mtl-isr1-slurm-gw`).
# It refreshes the repo, patches detect_live.py for CUDA, and ensures the env exists.
set -euo pipefail

REPO=/auto/swgwork1/eavidan/hackathon
ENV_NAME=cshack

echo "[1/5] sync repo (hard reset to origin/main — slurm copy is run-area only)"
if [ -d "$REPO/.git" ]; then
  cd "$REPO"
  git fetch origin
  git reset --hard origin/main
else
  cd /auto/swgwork1/eavidan
  git clone https://github.com/eden-avidan/CSHACK-2026-LKP.git hackathon
  cd "$REPO"
fi

echo "[2/5] patch detect_live.py for slurm (cuda + bundled clip + yolo26x + imgsz 1280)"
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("figure_recognition/detect_live.py")
s = p.read_text()
def sub(pattern, repl, label):
    global s
    if not re.search(pattern, s, flags=re.MULTILINE):
        raise SystemExit(f"line not found: {label}")
    s = re.sub(pattern, repl, s, count=1, flags=re.MULTILINE)
sub(r'^SOURCE\s*=.*$',     'SOURCE      = HERE / "samples" / "drone_test.mp4"', "SOURCE")
sub(r'^MODEL_NAME\s*=.*$', 'MODEL_NAME  = "yolo26x.pt"',                        "MODEL_NAME")
sub(r'^DEVICE\s*=.*$',     'DEVICE          = "cuda"',                          "DEVICE")
sub(r'^INFER_SIZE\s*=.*$', 'INFER_SIZE      = 1280',                            "INFER_SIZE")
p.write_text(s)
print("patched (idempotent).")
PY

echo "[3/5] check conda env"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  env '$ENV_NAME' exists"
else
  echo "  creating env '$ENV_NAME' from environment.yaml (CUDA)"
  # /labhome is full; redirect pip + conda caches to /auto/swgwork1
  export PIP_CACHE_DIR=/auto/swgwork1/eavidan/.cache/pip
  export TMPDIR=/auto/swgwork1/eavidan/.tmp
  export CONDA_PKGS_DIRS=/auto/swgwork1/eavidan/.conda/pkgs
  mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS"
  conda env create -f environment.yaml
fi

echo "[4/5] check yolo26x weights"
mkdir -p figure_recognition/models
if [ -f figure_recognition/models/yolo26x.pt ]; then
  echo "  yolo26x.pt already present"
else
  echo "  yolo26x.pt missing — run scp from your Mac (see below) or it'll auto-download on first run"
fi

echo "[5/5] write sbatch wrapper"
NODELIST="${NODE:-hgx-isr1-pre-[15,16]}"
PARTITION="${PARTITION:-ISR1-PRE}"
RESERVATION="${RESERVATION:-nonRA}"
mkdir -p scripts
{
  echo '#!/usr/bin/env bash'
  echo '#SBATCH --job-name=cshack-detect'
  echo '#SBATCH --gres=gpu:1'
  echo "#SBATCH --partition=${PARTITION}"
  echo "#SBATCH --reservation=${RESERVATION}"
  echo "#SBATCH --nodelist=${NODELIST}"
  echo '#SBATCH --time=01:00:00'
  echo '#SBATCH --output=slurm-%j.out'
  echo '#SBATCH --error=slurm-%j.err'
  echo ''
  echo 'set -euo pipefail'
  echo 'cd /auto/swgwork1/eavidan/hackathon'
  echo 'source "$(conda info --base)/etc/profile.d/conda.sh"'
  echo 'conda activate cshack'
  echo 'echo "[host] $(hostname)  [port] 8000"'
  echo 'echo "[tunnel from mac] ssh -L 8000:$(hostname):8000 eavidan@mtl-isr1-slurm-gw"'
  echo 'python figure_recognition/detect_live.py'
} > scripts/slurm_run.sbatch
chmod +x scripts/slurm_run.sbatch
echo "  partition: ${PARTITION}   reservation: ${RESERVATION}   nodelist: ${NODELIST}"
echo "  override: NODE=<name> PARTITION=<p> RESERVATION=<r> bash /tmp/slurm_setup.sh"

echo "==================================================="
echo "setup complete."
echo "to run:    sbatch scripts/slurm_run.sbatch"
echo "or live:   srun --gres=gpu:1 --time=01:00:00 --pty python figure_recognition/detect_live.py"
echo "==================================================="

#!/usr/bin/env bash
# =============================================================================
#  setup.sh — One-command environment setup
#  Usage: bash setup.sh
# =============================================================================
set -e

PYTHON=${PYTHON:-python3.10}
VENV=".venv"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Multi-Camera Defect Tracking — Setup Script   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
echo "▸ Checking Python..."
if ! command -v $PYTHON &>/dev/null; then
  echo "  ✗ Python 3.10+ not found. Install from https://python.org"
  exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Found Python $PY_VER"

# ── Docker check ──────────────────────────────────────────────────────────────
echo "▸ Checking Docker..."
if ! command -v docker &>/dev/null; then
  echo "  ✗ Docker not found. Install from https://docker.com"
  exit 1
fi
echo "  ✓ Docker $(docker --version | awk '{print $3}' | tr -d ',')"

# ── Virtual environment ───────────────────────────────────────────────────────
echo "▸ Creating virtual environment..."
$PYTHON -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip -q
echo "  ✓ Virtual env at ./$VENV"

# ── Python dependencies ───────────────────────────────────────────────────────
echo "▸ Installing Python dependencies..."
pip install -r requirements.txt -q
echo "  ✓ Dependencies installed"

# ── Config file ───────────────────────────────────────────────────────────────
if [ ! -f configs/config.yaml ]; then
  echo "▸ Creating configs/config.yaml from example..."
  cp configs/config.example.yaml configs/config.yaml
  echo "  ✓ Edit configs/config.yaml before running"
else
  echo "▸ configs/config.yaml already exists — skipping"
fi

# ── Models directory ──────────────────────────────────────────────────────────
mkdir -p models
if [ ! -f models/defect_yolov8m.pt ]; then
  echo ""
  echo "  ⚠  No model weights found at models/defect_yolov8m.pt"
  echo "     Download a pretrained model or train your own:"
  echo "     yolo train data=your_data.yaml model=yolov8m.pt epochs=100"
  echo ""
fi

# ── Logs directory ────────────────────────────────────────────────────────────
mkdir -p logs

# ── Kafka topic setup helper ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                 NEXT STEPS                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  1. Start infrastructure:"
echo "     docker-compose up -d zookeeper kafka postgres redis"
echo ""
echo "  2. Start central server:"
echo "     source .venv/bin/activate"
echo "     python server/main.py"
echo ""
echo "  3. Start API:"
echo "     uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  4. Start edge node (per camera):"
echo "     python edge/main.py --camera-id cam_01 --source 0"
echo ""
echo "  5. Open dashboard:"
echo "     open dashboard/index.html"
echo "     (or serve with: python -m http.server 3000 --directory dashboard)"
echo ""
echo "  6. API docs:"
echo "     http://localhost:8000/docs"
echo ""

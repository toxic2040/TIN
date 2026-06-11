#!/usr/bin/env bash
# epyc_setup.sh — One-shot idempotent provisioning for 96-core EPYC revalidation.
#
# Usage:  bash runs/epyc_setup.sh
#
# Installs Python 3.13 via miniconda, clones repo, installs deps, fetches
# SPICE kernels, runs tests. Safe to re-run.

set -euo pipefail

WORK_DIR="$HOME/TIN"

echo "=== TIN EPYC Revalidation — Server Setup ==="
echo "Start: $(date)"
echo ""

# ---- 0. System packages ----
echo "[0/7] System packages..."
apt-get update -qq
apt-get install -y -qq git curl build-essential sqlite3 > /dev/null 2>&1
echo "  Done."

# ---- 1. Miniconda + Python 3.13 ----
if ! command -v conda &> /dev/null; then
    echo "[1/7] Installing Miniconda..."
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
    conda create -y -n tin python=3.13 -q
    conda activate tin
else
    echo "[1/7] Conda exists, activating..."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate tin 2>/dev/null || conda create -y -n tin python=3.13 -q && conda activate tin
fi
echo "  Python: $(python --version)"

# ---- 2. Clone or update repo ----
if [ -d "$WORK_DIR/.git" ]; then
    echo "[2/7] Repo exists, pulling latest..."
    cd "$WORK_DIR"
    git pull --ff-only
else
    echo "[2/7] Cloning repo..."
    git clone https://github.com/toxic2040/TIN.git "$WORK_DIR"
    cd "$WORK_DIR"
fi

# ---- 3. Install (editable, dev extras) ----
echo "[3/7] Installing package..."
pip install -e ".[dev]" --quiet 2>&1 | tail -3

# ---- 4. Fetch SPICE kernels ----
KERNEL_DIR="$WORK_DIR/data/kernels"
mkdir -p "$KERNEL_DIR"

NAIF_BASE="https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
declare -A KERNELS=(
    ["de440s.bsp"]="$NAIF_BASE/spk/planets/de440s.bsp"
    ["naif0012.tls"]="$NAIF_BASE/lsk/naif0012.tls"
    ["pck00011.tpc"]="$NAIF_BASE/pck/pck00011.tpc"
)

for fname in "${!KERNELS[@]}"; do
    if [ ! -f "$KERNEL_DIR/$fname" ]; then
        echo "  Fetching $fname..."
        curl -sL "${KERNELS[$fname]}" -o "$KERNEL_DIR/$fname"
    else
        echo "  $fname already present."
    fi
done
echo "[4/7] SPICE kernels ready."

# ---- 5. Extract CRAWDAD traces ----
TRACE_DIR="$WORK_DIR/data/traces"
mkdir -p "$TRACE_DIR"
for archive in "$TRACE_DIR"/*.tar.gz; do
    [ -f "$archive" ] || continue
    base="$(basename "$archive" .tar.gz)"
    if [ ! -d "$TRACE_DIR/$base" ]; then
        echo "  Extracting $base..."
        tar -xzf "$archive" -C "$TRACE_DIR"
    fi
done
echo "[5/7] Trace data ready."

# ---- 6. Run tests ----
echo "[6/7] Running test suite..."
cd "$WORK_DIR"
if python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5; then
    echo "  Tests PASSED."
else
    echo "  WARNING: Tests had failures. Check output."
fi

# ---- 7. System info ----
echo "[7/7] System info:"
echo "  Python:    $(python --version 2>&1)"
echo "  CPU cores: $(nproc)"
echo "  RAM:       $(free -h | awk '/Mem:/{print $2}')"
echo "  Disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "  Hostname:  $(hostname)"
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  conda activate tin"
echo "  cd ~/TIN"
echo "  nohup python runs/epyc_phase1.py > phase1.log 2>&1 &"
echo "  tail -f phase1.log"

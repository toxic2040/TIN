#!/usr/bin/env bash
# epyc_setup_v2.sh — Provision EPYC server for Phase 3 experiments.
#
# Usage:  bash runs/epyc_setup_v2.sh
#
# Installs Python 3.13 via miniconda, clones repo, installs deps, fetches
# SPICE kernels, downloads vehicular GPS data, runs tests.  Safe to re-run.

set -eo pipefail

WORK_DIR="$HOME/TIN"

echo "=== TIN EPYC Phase 3 — Server Setup ==="
echo "Start: $(date)"
echo ""

# ---- 0. System packages ----
echo "[0/8] System packages..."
apt-get update -qq
apt-get install -y -qq git curl build-essential sqlite3 wget > /dev/null 2>&1
echo "  Done."

# ---- 1. Miniconda + Python 3.13 ----
if ! command -v conda &> /dev/null; then
    echo "[1/8] Installing Miniconda..."
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
    conda create -y -n tin python=3.13 -q
    conda activate tin
else
    echo "[1/8] Conda exists, activating..."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate tin 2>/dev/null || (conda create -y -n tin python=3.13 -q && conda activate tin)
fi
echo "  Python: $(python --version)"

# ---- 2. Clone or update repo ----
if [ -d "$WORK_DIR/.git" ]; then
    echo "[2/8] Repo exists, pulling latest..."
    cd "$WORK_DIR"
    git pull --ff-only
else
    echo "[2/8] Cloning repo..."
    git clone https://github.com/toxic2040/TIN.git "$WORK_DIR"
    cd "$WORK_DIR"
fi

# ---- 3. Install (editable, dev extras) ----
echo "[3/8] Installing package..."
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
echo "[4/8] SPICE kernels ready."

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
echo "[5/8] Trace data ready."

# ---- 6. Download vehicular GPS data ----
echo "[6/8] Downloading vehicular GPS data..."
VEH_DIR="$WORK_DIR/data/vehicular"
mkdir -p "$VEH_DIR"

VEHICULAR_OK=false

# Option A: SF cabspotting GPS data (verified working URL — 91 MB, 537 cabs, 24 days)
SFCAB_DIR="$VEH_DIR/sfcabs"
if [ -d "$SFCAB_DIR/cabspottingdata" ] && [ "$(ls "$SFCAB_DIR/cabspottingdata/"*.txt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  SF cab data already present."
    VEHICULAR_OK=true
else
    mkdir -p "$SFCAB_DIR"
    SFCAB_URL="https://github.com/PDXostc/rvi_big_data/raw/master/cabspottingdata.tar.gz"
    echo "  Downloading SF cabspotting GPS data (91 MB)..."
    echo "    URL: $SFCAB_URL"
    if curl -sL --connect-timeout 30 --max-time 300 "$SFCAB_URL" -o "$SFCAB_DIR/cabspottingdata.tar.gz"; then
        if [ -s "$SFCAB_DIR/cabspottingdata.tar.gz" ]; then
            echo "  Extracting..."
            cd "$SFCAB_DIR" && tar -xzf cabspottingdata.tar.gz && cd "$WORK_DIR"
            N_FILES=$(ls "$SFCAB_DIR/cabspottingdata/"*.txt 2>/dev/null | wc -l)
            echo "  Extracted $N_FILES cab files"
            if [ "$N_FILES" -gt 0 ]; then
                VEHICULAR_OK=true
            fi
        fi
    else
        echo "  Download failed."
    fi
fi

# Option B: Generate synthetic vehicular contacts as fallback
if [ "$VEHICULAR_OK" = false ]; then
    echo "  Automatic download failed. Generating synthetic vehicular contacts..."
    python3 -c "
import json, random, math

random.seed(42)
N_VEHICLES = 100
AREA_M = 5000.0
RANGE_M = 200.0
SPEED_MIN, SPEED_MAX = 8.0, 17.0  # m/s (30-60 km/h)
DT = 10.0  # seconds per step
DURATION_H = 24.0
N_STEPS = int(DURATION_H * 3600 / DT)

# Initialise vehicles: (x, y, vx, vy)
vehicles = []
for _ in range(N_VEHICLES):
    x = random.uniform(0, AREA_M)
    y = random.uniform(0, AREA_M)
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(SPEED_MIN, SPEED_MAX)
    vehicles.append([x, y, speed * math.cos(angle), speed * math.sin(angle)])

contacts = []
active = {}  # (i,j) -> start_time

for step in range(N_STEPS):
    t = step * DT

    # Move vehicles (wrap around — toroidal boundary)
    for v in vehicles:
        v[0] = (v[0] + v[2] * DT) % AREA_M
        v[1] = (v[1] + v[3] * DT) % AREA_M
        # Random direction change (10% chance per step)
        if random.random() < 0.10:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(SPEED_MIN, SPEED_MAX)
            v[2] = speed * math.cos(angle)
            v[3] = speed * math.sin(angle)

    # Check pairwise distances
    in_range = set()
    for i in range(N_VEHICLES):
        for j in range(i + 1, N_VEHICLES):
            dx = min(abs(vehicles[i][0] - vehicles[j][0]),
                     AREA_M - abs(vehicles[i][0] - vehicles[j][0]))
            dy = min(abs(vehicles[i][1] - vehicles[j][1]),
                     AREA_M - abs(vehicles[i][1] - vehicles[j][1]))
            if math.sqrt(dx*dx + dy*dy) <= RANGE_M:
                in_range.add((i, j))

    # Close contacts that ended
    for pair in list(active.keys()):
        if pair not in in_range:
            start = active.pop(pair)
            dur = t - start
            if dur >= 1.0:
                i, j = pair
                for fn, tn in [(str(i), str(j)), (str(j), str(i))]:
                    contacts.append({
                        'from_node': fn, 'to_node': tn,
                        'start_s': start, 'duration_s': dur,
                        'latency_s': 0.1, 'p_success': 1.0,
                        'data_rate_kbps': 1000.0,
                    })

    # Open new contacts
    for pair in in_range:
        if pair not in active:
            active[pair] = t

# Close remaining
for pair, start in active.items():
    dur = N_STEPS * DT - start
    if dur >= 1.0:
        i, j = pair
        for fn, tn in [(str(i), str(j)), (str(j), str(i))]:
            contacts.append({
                'from_node': fn, 'to_node': tn,
                'start_s': start, 'duration_s': dur,
                'latency_s': 0.1, 'p_success': 1.0,
                'data_rate_kbps': 1000.0,
            })

contacts.sort(key=lambda c: c['start_s'])
nodes = set()
for c in contacts:
    nodes.add(c['from_node'])
    nodes.add(c['to_node'])
print(f'Generated {len(contacts)} contacts across {len(nodes)} vehicles')

# Save as haggle-like format for easy parsing
with open('$VEH_DIR/synthetic_vehicular_contacts.json', 'w') as f:
    json.dump({'contacts': contacts, 'type': 'synthetic_vehicular',
               'n_vehicles': N_VEHICLES, 'area_m': AREA_M, 'range_m': RANGE_M,
               'duration_h': DURATION_H}, f)
print('Saved synthetic vehicular contacts')
"
    if [ -f "$VEH_DIR/synthetic_vehicular_contacts.json" ]; then
        VEHICULAR_OK=true
        echo "  Synthetic vehicular contacts generated."
    fi
fi

if [ "$VEHICULAR_OK" = true ]; then
    echo "[6/8] Vehicular data ready."
else
    echo "[6/8] WARNING: No vehicular data available."
    echo "  Manual download required. See README at:"
    echo "    - Roma taxi:  https://ieee-dataport.org/open-access/crawdad-romataxis"
    echo "    - T-Drive:    https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/"
    echo "  Place GPS files in $VEH_DIR/{roma,tdrive}/"
fi

# ---- 7. Run tests ----
echo "[7/8] Running test suite..."
cd "$WORK_DIR"
if python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5; then
    echo "  Tests PASSED."
else
    echo "  WARNING: Tests had failures. Check output."
fi

# ---- 8. System info ----
echo "[8/8] System info:"
echo "  Python:    $(python --version 2>&1)"
echo "  CPU cores: $(nproc)"
echo "  RAM:       $(free -h | awk '/Mem:/{print $2}')"
echo "  Disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "  Hostname:  $(hostname)"
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. SCP experiment scripts from local machine:"
echo "     scp runs/run_routing_independence.py runs/run_vehicular_gamma.py runs/epyc_phase3.py root@\$(hostname):~/TIN/runs/"
echo ""
echo "  2. Run Phase 3:"
echo "     conda activate tin"
echo "     cd ~/TIN"
echo "     nohup python runs/epyc_phase3.py > phase3.log 2>&1 &"
echo "     tail -f phase3.log"

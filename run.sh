#!/usr/bin/env bash
# RoomSync — start everything.
# Usage:  ./run.sh [room-code] [host|peer]
# Example: ./run.sh ABC123 host

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
ROOM_CODE="${1:-DEMO01}"
ROLE="${2:-peer}"

export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_LIBRARY_PATH:-}"

# ── Python venv ───────────────────────────────────────────────────────────────
if [ ! -d "$REPO/.venv" ]; then
    echo "[run.sh] Creating Python venv..."
    python3 -m venv "$REPO/.venv"
fi
source "$REPO/.venv/bin/activate"
pip install --quiet -r "$REPO/requirements.txt"

# ── C++ pipeline (build if needed) ───────────────────────────────────────────
if [ ! -f "$REPO/build/pipeline/roomsync_test" ]; then
    echo "[run.sh] Building C++ pipeline..."
    cmake -S "$REPO" -B "$REPO/build" -DCMAKE_BUILD_TYPE=Release \
          -DWITH_PORTAUDIO=OFF -DWITH_OPUS=OFF -DWITH_LIBSODIUM=OFF \
          -DWITH_ONNXRUNTIME=OFF -DWITH_MAXINE=OFF
    cmake --build "$REPO/build" --parallel "$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
fi

# ── Rust/Node toolchain ───────────────────────────────────────────────────────
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# ── Start coordinator in background ──────────────────────────────────────────
echo "[run.sh] Starting coordinator (room=$ROOM_CODE role=$ROLE)..."
cd "$REPO/coordinator"
python3 coordinator.py --room-code "$ROOM_CODE" --role "$ROLE" &
COORD_PID=$!
cd "$REPO"

cleanup() {
    echo "[run.sh] Shutting down..."
    kill "$COORD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Launch desktop app ────────────────────────────────────────────────────────
echo "[run.sh] Launching RoomSync desktop app..."
if [ -f "$REPO/app/src-tauri/target/release/RoomSync" ]; then
    "$REPO/app/src-tauri/target/release/RoomSync"
else
    cd "$REPO/app" && npx tauri dev
fi

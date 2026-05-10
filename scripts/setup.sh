#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

info()  { echo "[roomsync] $*"; }
error() { echo "[roomsync] ERROR: $*" >&2; exit 1; }

OS="$(uname -s)"
info "Detected OS: $OS"

# ── System dependencies ──────────────────────────────────────────────────────
install_system_deps() {
    case "$OS" in
    Darwin)
        command -v brew >/dev/null || error "Homebrew not found — install from https://brew.sh"
        brew install cmake python@3.11 portaudio opus libsodium pkg-config
        ;;
    Linux)
        if command -v apt >/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y cmake python3.11 python3.11-venv python3-pip \
                portaudio19-dev libopus-dev libsodium-dev libasound2-dev \
                build-essential pkg-config
        elif command -v dnf >/dev/null; then
            sudo dnf install -y cmake python3.11 portaudio-devel opus-devel \
                libsodium-devel alsa-lib-devel gcc-c++ pkgconfig
        else
            error "Unsupported Linux distro — install cmake, portaudio, opus, libsodium manually"
        fi
        ;;
    *)
        error "Unsupported OS: $OS. See BUILDPLAN.md §4.1 for Windows instructions."
        ;;
    esac
}

# ── Rust / Cargo (for Tauri) ──────────────────────────────────────────────────
install_rust() {
    if ! command -v cargo >/dev/null; then
        info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
        source "$HOME/.cargo/env"
    fi
    cargo install tauri-cli --version "^1" --locked 2>/dev/null || true
}

# ── Node.js (for Tauri frontend) ──────────────────────────────────────────────
install_node() {
    if ! command -v node >/dev/null; then
        if command -v brew >/dev/null; then
            brew install node
        else
            error "Node.js not found — install from https://nodejs.org"
        fi
    fi
}

# ── Python training environment ───────────────────────────────────────────────
setup_python_env() {
    PYTHON="python3.11"
    command -v "$PYTHON" >/dev/null || PYTHON="python3"
    info "Setting up Python training environment..."

    cd "$ROOT/training"
    "$PYTHON" -m venv venv
    source venv/bin/activate

    pip install --upgrade pip -q
    pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q  # CPU for dev; use cu118 for GPU
    pip install numpy scipy librosa soundfile webrtcvad pesq pystoi \
                matplotlib tqdm wandb opuslib aiohttp zeroconf cryptography -q

    deactivate
    info "Python environment ready at training/venv"
}

# ── C++ pipeline build ────────────────────────────────────────────────────────
build_pipeline() {
    info "Building C++ pipeline (CPU-only, debug)..."
    cd "$ROOT"
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Debug \
        -DWITH_MAXINE=OFF \
        -DWITH_PORTAUDIO=ON \
        -DWITH_OPUS=ON \
        -DBUILD_TESTS=ON
    cmake --build build --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
    info "Build output: build/roomsync_test"
}

# ── Desktop app deps ──────────────────────────────────────────────────────────
setup_app() {
    info "Installing frontend dependencies..."
    cd "$ROOT/app"
    npm install --silent
    info "App deps installed. Run 'npm run tauri:dev' to start."
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    info "=== RoomSync setup ==="
    install_system_deps
    install_rust
    install_node
    setup_python_env
    build_pipeline
    setup_app

    info ""
    info "=== Setup complete ==="
    info "Next steps:"
    info "  1. Download datasets:   cd training/datasets && ./download_aec.sh --dev"
    info "  2. Train NRES (fast):   cd training/nres && python train.py --epochs 10 --fast-dev"
    info "  3. Run pipeline test:   ./build/roomsync_test"
    info "  4. Launch app:          cd app && npm run tauri:dev"
}

main "$@"

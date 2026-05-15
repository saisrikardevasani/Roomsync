# RoomSync

> **Status: 🚧 In Active Development** — This is a research and learning build being actively implemented. The full end-to-end system is not yet production-ready; the sections below document the architecture and what is built so far.

---

## What's implemented today

- **C++ audio pipeline** — AEC (512-tap NLMS), Noise Suppression (spectral subtraction + VAD-gated PSD), PLC (LPC synthesis), Source Separation (passthrough stub with Conv-TasNet hooks); pipeline orchestrator with FrameStats; 13 signal-level tests passing (ERLE 1.7 dB → 10.4 dB at steady state)
- **Python coordinator** — mDNS peer discovery (Zeroconf), reference audio bus (far-end speaker → AEC input), Opus relay between peers
- **Encryption modules** — Signal Protocol Double Ratchet session (AES-256-GCM per frame), MLS group key skeleton (RFC 9420)
- **Tauri desktop app shell** — React + TypeScript UI, Rust bridge with Tauri IPC commands, pipeline lifecycle management, FrameStats polling
- **NVIDIA Maxine GPU path** — compile-time opt-in with transparent CPU fallback
- **Test scaffolding** — C++ and Python test suites wired into CMake

## In progress / next milestones

- End-to-end audio loop validation on real hardware
- NLMS double-talk robustness tuning under simultaneous speech
- ONNX model integration: LPCNet (PLC) and Conv-TasNet (separation)
- Multi-peer relay reliability and MLS epoch rotation testing
- Tauri UI metrics wired to live pipeline stats

---

## What it does

- **Acoustic Echo Cancellation (AEC)** — 512-tap NLMS adaptive filter converges to the local acoustic echo path in real time, delivering 10+ dB ERLE at steady state
- **Noise Suppression** — Spectral subtraction (Boll 1979) with per-bin Wiener gain, over-subtraction, and a VAD-gated noise PSD estimator
- **Packet Loss Concealment (PLC)** — Tiered: LPC synthesis (always available) → LPCNet ONNX model (optional) for AI reconstruction
- **Source Separation** — Passthrough stub; hooks for a trained Conv-TasNet ONNX model
- **GPU Acceleration** — Optional NVIDIA Maxine path; falls back to CPU pipeline transparently
- **End-to-End Encryption** — Double Ratchet (Signal Protocol) per frame + MLS group key rotation for multi-party sessions
- **Zero-config discovery** — mDNS/Zeroconf peer discovery on the local network; no server required

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Tauri Desktop App  (React + TypeScript)                 │
│  • Room code join/leave                                  │
│  • Live metrics: latency, ERLE, noise level, packet loss │
└──────────────┬───────────────────────────────────────────┘
               │ Tauri commands (IPC)
┌──────────────▼───────────────────────────────────────────┐
│  Rust bridge  (src-tauri)                                │
│  • Manages pipeline lifecycle                            │
│  • Polls and serialises FrameStats every 250 ms          │
└──────────────┬───────────────────────────────────────────┘
               │ FFI / in-process
┌──────────────▼───────────────────────────────────────────┐
│  C++ Audio Pipeline  (pipeline/)                         │
│  Stage 1 → AECProcessor   (NLMS, optional ONNX NRES)    │
│  Stage 2 → NoiseSuppressor (spectral subtraction + FFT)  │
│  Stage 3 → SourceSeparator (passthrough / ONNX model)   │
│  Stage 4 → PLCProcessor   (LPC synthesis / LPCNet ONNX) │
└──────────────┬───────────────────────────────────────────┘
               │ asyncio / WebSocket
┌──────────────▼───────────────────────────────────────────┐
│  Python Coordinator  (coordinator/)                      │
│  • mDNS peer discovery (Zeroconf)                        │
│  • Reference audio bus (far-end speaker → AEC)           │
│  • Opus relay between peers                              │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│  Encryption layer  (encryption/)                         │
│  • Signal Protocol: Double Ratchet, AES-256-GCM          │
│  • MLS RFC 9420: epoch key rotation on member changes    │
└──────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
RoomSync/
├── pipeline/               # C++ audio pipeline (CMake)
│   ├── aec/                # Acoustic echo cancellation (NLMS + optional NRES)
│   ├── noise/              # Noise suppression (spectral subtraction)
│   ├── plc/                # Packet loss concealment (LPC + optional LPCNet)
│   ├── separation/         # Source separation stub
│   ├── pipeline.cpp/.h     # Pipeline orchestrator
│   └── config.h            # Shared constants (FRAME_SIZE, SAMPLE_RATE …)
├── maxine/                 # NVIDIA Maxine GPU path + CPU fallback
├── app/                    # Tauri desktop application
│   ├── src/                # React + TypeScript UI
│   └── src-tauri/          # Rust backend (Tauri commands, pipeline bridge)
├── coordinator/            # Python async coordinator
│   ├── coordinator.py      # Entry point
│   ├── discovery.py        # mDNS peer discovery
│   ├── reference_bus.py    # Far-end audio distribution
│   └── relay.py            # Peer-to-peer Opus relay
├── encryption/
│   ├── signal/session.py   # Double Ratchet session
│   └── mls/group.py        # MLS group key management
├── training/               # ML training scripts
│   ├── nres/               # Non-linear residual echo suppression (ONNX)
│   ├── noise/              # RNNoise-style noise suppressor
│   ├── lpcnet/             # LPCNet PLC model
│   ├── separation/         # Conv-TasNet source separator
│   └── datasets/           # Dataset download scripts
├── tests/                  # C++ and Python test suites
├── scripts/setup.sh        # One-command environment setup
├── run.sh                  # Launch coordinator + desktop app
└── requirements.txt        # Python dependencies
```

---

## Quick start

### Prerequisites

| Tool | Version |
|------|---------|
| CMake | ≥ 3.20 |
| C++ compiler | Clang 14+ or GCC 12+ (C++17) |
| Rust + Cargo | ≥ 1.70 (via [rustup](https://rustup.rs)) |
| Node.js | ≥ 18 |
| Python | ≥ 3.9 |

### 1. Clone

```bash
git clone https://github.com/saisrikardevasani/Roomsync.git
cd Roomsync
```

### 2. Build the C++ pipeline

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run the unit tests (13 signal-level tests):

```bash
./build/pipeline/roomsync_test
```

### 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the desktop app (dev mode)

```bash
cd app
npm install
npm run tauri dev
```

### 5. (Optional) Launch the coordinator

```bash
source .venv/bin/activate
python coordinator/coordinator.py
```

Or use the convenience script which starts both:

```bash
./run.sh
```

---

## Optional: ONNX model acceleration

Each pipeline stage can be upgraded with a trained ONNX model. Place models at the paths configured in `PipelineConfig` and rebuild with:

```bash
cmake -B build -DROOMSYNC_WITH_ONNXRUNTIME=ON \
      -DOnnxRuntime_ROOT=/path/to/onnxruntime
cmake --build build --parallel
```

Training scripts for each model are in `training/`.

---

## Optional: NVIDIA Maxine GPU pipeline

```bash
cmake -B build -DROOMSYNC_WITH_MAXINE=ON \
      -DMaxine_ROOT=/path/to/maxine-sdk
cmake --build build --parallel
```

When `use_gpu: true` is set at runtime the Maxine path handles AEC + noise suppression entirely on GPU; the CPU pipeline is the fallback.

---

## Signal-level test results

| Test | Result |
|------|--------|
| Zero input → near-zero output | PASS |
| NLMS converges: ERLE(frame 10) < ERLE(frame 199) | PASS — 1.7 dB → 10.4 dB |
| Noise reduction > 0 dB after training | PASS — 1.8 dB |
| LPC PLC has non-zero energy over 3 lost frames | PASS — 0.80 RMS |
| p99 pipeline latency < 10 ms | PASS — 0.096 ms |
| Stats clear to zero after Pipeline::reset() | PASS |

---

## License

MIT

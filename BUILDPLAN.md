# RoomSync — Build Plan

> **What:** A desktop app that eliminates echo, reconstructs lost speech, and encrypts audio end-to-end for multi-laptop same-room meetings.
> **Stack:** Python (training) · C/C++ (real-time audio pipeline) · Electron or Tauri (desktop shell) · NVIDIA Maxine SDK (GPU acceleration)
> **Timeline:** 12 weeks · Team of 4

---

## Table of Contents

1. [Problem Recap](#1-problem-recap)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Dataset Acquisition](#5-dataset-acquisition)
6. [Model Training](#6-model-training)
7. [Core Pipeline Implementation](#7-core-pipeline-implementation)
8. [Room Coordination Layer](#8-room-coordination-layer)
9. [Encryption Layer](#9-encryption-layer)
10. [NVIDIA Maxine Integration](#10-nvidia-maxine-integration)
11. [Desktop App & Virtual Audio Device](#11-desktop-app--virtual-audio-device)
12. [Testing & Benchmarking](#12-testing--benchmarking)
13. [Week-by-Week Milestones](#13-week-by-week-milestones)
14. [Success Metrics](#14-success-metrics)

---

## 1. Problem Recap

Three failures happen simultaneously when multiple laptops share one room on a video call:

| # | Failure | Root Cause | Existing Fix | Why it Breaks |
|---|---------|-----------|-------------|---------------|
| P1 | Acoustic echo loops | Each mic picks up other laptops' speakers | Zoom AEC3 | Fails above 15% double-talk; assumes 1 laptop per room |
| P2 | Voice cross-talk | Same voice arrives at multiple mics at different delays | Mute everyone | Kills natural conversation |
| P3 | Speech permanently lost | UDP packet drops; Opus FEC only covers 20ms | None | 100–300ms drops common on Wi-Fi |
| P4 | No audio privacy | Platform holds decryption keys | None | Sensitive hiring/legal conversations exposed |

**RoomSync solves all four in a single <10ms pipeline.**

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ROOMSYNC APP                         │
│                                                             │
│  [Mic Input]                                                │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────────────────────┐        │
│  │  AUDIO PIPELINE  (target: <10ms total)          │        │
│  │                                                 │        │
│  │  1. AEC  ←── room reference bus                │        │
│  │     (WebRTC AEC3 + neural residual suppressor)  │        │
│  │     Target: ERLE >45dB, <8ms                   │        │
│  │          │                                      │        │
│  │  2. Noise Suppression                           │        │
│  │     (RNNoise fork / Maxine NvAFX_DENOISER)     │        │
│  │     Target: -15dB, <3ms                        │        │
│  │          │                                      │        │
│  │  3. Source Separation                           │        │
│  │     (detect & suppress cross-laptop voices)     │        │
│  │          │                                      │        │
│  │  4. Packet Loss Concealment                     │        │
│  │     (Opus RED FEC → LPCNet for >40ms loss)     │        │
│  │     Target: recover up to 100ms, <10ms         │        │
│  │          │                                      │        │
│  │  5. Encrypt                                     │        │
│  │     (AES-256-GCM per 10ms frame)               │        │
│  │     (Signal Protocol 1:1 / MLS RFC 9420 group) │        │
│  └─────────────────────────────────────────────────┘        │
│      │                                                      │
│      ▼                                                      │
│  [Virtual Audio Device] → Zoom / Teams / Meet               │
│                                                             │
│  ┌──────────────────────────────┐                           │
│  │  ROOM COORDINATOR            │                           │
│  │  - Room code pairing         │                           │
│  │  - Shared audio reference    │                           │
│  │  - Peer discovery (LAN/TURN) │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### GPU Acceleration Path (optional)

When NVIDIA GPU detected (CUDA driver ≥525), steps 1–3 offload to **NVIDIA Maxine Audio Effects SDK v3.0**:

```
CPU path:  AEC3 + NRES + RNNoise + LPCNet   →  <10ms
GPU path:  NvAFX_AEC + NvAFX_DENOISER + NvAFX_SUPER_RESOLUTION  →  <3ms
```

Fallback to CPU within 100ms if GPU driver crashes or VRAM exhausted.

---

## 3. Repository Structure

```
roomsync/
│
├── pipeline/                   # Core real-time audio pipeline (C++)
│   ├── aec/                    # Acoustic Echo Cancellation
│   │   ├── webrtc_aec3/        # WebRTC AEC3 (vendored)
│   │   └── nres/               # Neural Residual Echo Suppressor
│   ├── noise/                  # Noise suppression (RNNoise fork)
│   ├── separation/             # Multi-speaker source separation
│   ├── plc/                    # Packet Loss Concealment
│   │   ├── fec/                # Opus RED FEC (RFC 2198)
│   │   └── lpcnet/             # LPCNet AI reconstruction
│   └── pipeline.cpp            # Main pipeline orchestrator
│
├── encryption/                 # E2E encryption layer
│   ├── signal/                 # Signal Protocol (1:1)
│   └── mls/                    # MLS RFC 9420 (group)
│
├── coordinator/                # Room coordination service
│   ├── discovery.py            # LAN peer discovery
│   ├── relay.py                # TURN relay for non-LAN
│   └── reference_bus.py        # Shared audio reference bus
│
├── maxine/                     # NVIDIA Maxine GPU integration
│   ├── maxine_pipeline.cpp     # GPU pipeline wrapper
│   └── fallback.cpp            # CPU fallback logic
│
├── training/                   # Model training (Python)
│   ├── datasets/               # Dataset download + prep scripts
│   │   ├── download_aec.sh     # Microsoft AEC Challenge
│   │   ├── download_dns.sh     # Microsoft DNS-5
│   │   ├── download_plc.sh     # ICASSP 2024 PLC dataset
│   │   ├── download_rir.sh     # OpenSLR RIR database
│   │   └── download_librimix.sh # LibriheavyMix
│   ├── nres/                   # NRES model training
│   ├── lpcnet/                 # LPCNet PLC training
│   └── separation/             # Source separation training
│
├── virtual_device/             # Virtual audio driver
│   ├── macos/                  # CoreAudio HAL plugin
│   └── windows/                # WASAPI virtual device
│
├── app/                        # Desktop shell (Tauri + React)
│   ├── src/
│   └── src-tauri/
│
├── tests/                      # Benchmarks and integration tests
│   ├── latency/
│   ├── erle/
│   └── mos/
│
└── docs/
    ├── BUILDPLAN.md            # This file
    └── PRD.md
```

---

## 4. Environment Setup

### 4.1 System Dependencies

```bash
# macOS
brew install cmake python@3.11 portaudio opus libsodium

# Ubuntu / Debian
sudo apt install cmake python3.11 python3-pip portaudio19-dev \
  libopus-dev libsodium-dev libasound2-dev build-essential

# Windows (via winget)
winget install cmake python3.11 # then install vcpkg for portaudio, opus
```

### 4.2 Python Training Environment

```bash
cd roomsync/training
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy librosa soundfile webrtcvad pesq pystoi
pip install matplotlib tqdm wandb
```

### 4.3 C++ Pipeline Build

```bash
cd roomsync
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_MAXINE=OFF
make -j$(nproc)
```

To build with NVIDIA Maxine support:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_MAXINE=ON \
  -DMAXINE_SDK_PATH=/path/to/maxine-audio-effects-sdk-3.0
make -j$(nproc)
```

### 4.4 NVIDIA Maxine SDK Setup

1. Download **NVIDIA Maxine Audio Effects SDK v3.0** from [NVIDIA Developer](https://developer.nvidia.com/maxine-sdk-downloads)
2. Requires: CUDA ≥11.8, cuDNN ≥8.6, Driver ≥525
3. Set environment variable: `export MAXINE_SDK=/path/to/sdk`
4. APIs used:
   - `NvAFX_ACOUSTIC_ECHO_CANCELLATION`
   - `NvAFX_DENOISER`
   - `NvAFX_SUPER_RESOLUTION` (8kHz → 16kHz bandwidth extension)

---

## 5. Dataset Acquisition

Run all downloads in parallel — total ~300GB across all datasets.

### 5.1 Microsoft AEC Challenge (Echo Cancellation)

```bash
# 50,000 real recordings + 10,000 synthetic scenarios
git clone https://github.com/microsoft/AEC-Challenge
cd AEC-Challenge
python download_data.py --split train
```

**What you get:**
- Real: 50,000 recordings from 10,000+ devices and environments
- Synthetic: 10,000 scenarios — single talk, double talk, near/far noise, non-linear distortion
- Format: 16kHz, 16-bit WAV

### 5.2 Microsoft DNS Challenge 5 (Noise Suppression)

```bash
git clone https://github.com/microsoft/DNS-Challenge
cd DNS-Challenge
# Downloads clean speech + 150+ noise categories + RIRs
python download-dns-challenge-5.py
```

**What you get:**
- Clean speech from LibriVox (500+ hours)
- 150+ noise categories: keyboard, HVAC, crowd, traffic, construction
- Room impulse responses for convolution augmentation
- Format: 16kHz / 48kHz, 16-bit WAV

### 5.3 LibriheavyMix (Multi-Speaker Separation)

```bash
# 20,000 hours of reverberant overlapping speech
# Paper: https://arxiv.org/abs/2409.00819
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ltnghia/LibriheavyMix', repo_type='dataset', local_dir='./data/libriheavymix')
"
```

**What you get:**
- 20,000 hours of 2–4 speaker overlapping reverberant speech
- Speaker turn annotations for diarization
- Transcripts with punctuation and casing
- Consistent room acoustics per session (same RIR per session, simulating one physical room)

### 5.4 ICASSP 2024 PLC Challenge (Packet Loss Concealment)

```bash
git clone https://github.com/microsoft/PLC-Challenge
cd PLC-Challenge
# Register at https://aka.ms/plc_challenge to get download token
python download_data.py --token YOUR_TOKEN --year 2024
```

**What you get:**
- Clean speech + lossy versions with real-world packet loss patterns
- LibriVox conversational speech (permissive license)
- Lost-packet annotations (exact timestamps of dropped frames)
- Blind test set without references (for evaluation)

### 5.5 OpenSLR Room Impulse Response Database

```bash
# 60,000+ simulated + real RIRs at 16kHz
wget https://www.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip -d ./data/rir
```

**What you get:**
- Real RIRs: offices, meeting rooms, large rooms
- Simulated RIRs: various geometries via image source method
- Isotropic and point-source noise recordings
- Format: 16kHz, 16-bit WAV

### Dataset Summary

| Dataset | Use | Size | License |
|---------|-----|------|---------|
| Microsoft AEC Challenge | Train AEC model | ~50GB | CC-BY-4.0 |
| Microsoft DNS-5 | Train noise suppressor | ~500GB | CC-BY-4.0 |
| LibriheavyMix | Train speaker separator | ~200GB | CC-BY-4.0 |
| ICASSP 2024 PLC | Train LPCNet PLC | ~20GB | Research (register) |
| OpenSLR RIR | Room simulation augmentation | ~10GB | Apache 2.0 |

---

## 6. Model Training

### 6.1 Neural Residual Echo Suppressor (NRES)

The NRES is a 4-layer DCCRN (Deep Complex Convolutional Recurrent Network) that sits on top of WebRTC AEC3. AEC3 removes the linear echo component; NRES removes what's left (non-linear distortion, residual double-talk artifacts).

**Architecture:**
- Input: mic signal + AEC3 residual (complex spectrogram, 257 bins, 16kHz)
- 4 encoder layers (complex conv), 2 LSTM layers, 4 decoder layers
- 1.2M parameters
- Output: clean speech mask applied to residual

**Training:**

```bash
cd training/nres

# Step 1: Prepare training data from AEC Challenge
python prepare_data.py \
  --aec_dir ../../data/AEC-Challenge \
  --rir_dir ../../data/rir \
  --output_dir ./prepared \
  --n_synthetic 10000 \
  --n_real 50000

# Step 2: Train NRES
python train.py \
  --data_dir ./prepared \
  --model dccrn \
  --layers 4 \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-3 \
  --loss sisnr+mse \
  --checkpoint_dir ./checkpoints \
  --wandb_project roomsync-nres

# Step 3: Export to ONNX for C++ runtime
python export_onnx.py \
  --checkpoint ./checkpoints/best.pt \
  --output ../../pipeline/aec/nres/nres.onnx
```

**Validation targets:**
- ERLE >45dB on AEC Challenge test set
- MOS >4.0 on double-talk scenarios
- Inference latency <4ms on Intel i7-1265U (single core)

### 6.2 RNNoise Noise Suppressor

Based on Mozilla's RNNoise but extended with a 3-layer GRU and trained on DNS-5.

```bash
cd training/noise

# Prepare mixed training data: speech + noise at SNR -5 to 20dB
python mix_dataset.py \
  --speech_dir ../../data/DNS-5/clean \
  --noise_dir ../../data/DNS-5/noise \
  --rir_dir ../../data/rir \
  --output_dir ./mixed \
  --snr_range -5 20 \
  --n_samples 200000

# Train RNNoise fork
python train_rnnoise.py \
  --data_dir ./mixed \
  --gru_layers 3 \
  --hidden_size 96 \
  --epochs 150 \
  --output_dir ./checkpoints

# Export to C weights file (RNNoise uses C codegen)
python export_weights.py \
  --checkpoint ./checkpoints/best.pt \
  --output ../../pipeline/noise/rnnoise_weights.c
```

**Validation targets:**
- -15dB noise reduction on DNS-5 test set
- No speech distortion (SI-SDR >18dB)
- Inference <3ms on ARM Cortex-A76

### 6.3 LPCNet Packet Loss Concealment

Reconstructs up to 100ms of lost speech from prior context. Uses Mozilla's LPCNet as the base.

```bash
cd training/lpcnet

# Prepare loss simulation data from ICASSP 2024 PLC dataset
python simulate_loss.py \
  --plc_dir ../../data/PLC-2024 \
  --loss_patterns ./patterns/real_wifi_loss.json \
  --output_dir ./prepared \
  --max_loss_ms 100 \
  --n_samples 50000

# Train LPCNet PLC
python train_lpcnet.py \
  --data_dir ./prepared \
  --model_size small \    # 0.8M params — must stay under 5ms inference
  --epochs 200 \
  --lr 2e-4 \
  --output_dir ./checkpoints

# Validate on INTERSPEECH 2022 blind test set
python evaluate.py \
  --checkpoint ./checkpoints/best.pt \
  --test_dir ../../data/PLC-2022/blind_test

# Export ONNX
python export_onnx.py \
  --checkpoint ./checkpoints/best.pt \
  --output ../../pipeline/plc/lpcnet/lpcnet.onnx
```

**Validation targets:**
- MOS >4.0 on reconstructed 60ms loss segments
- PESQ >3.5 on ICASSP 2024 blind test
- Inference <5ms on ARM (measured on Pixel 6)
- Graceful on non-speech: no artifacts on music or DTMF tones

### 6.4 Multi-Speaker Source Separator

Detects which voices in the mic stream are coming from other laptops in the same room (cross-talk) and suppresses them, while preserving the intended near-end speaker.

```bash
cd training/separation

# Prepare LibriheavyMix for room-aware training
# Key: simulate "2 people in same room, each on laptop" scenario
python prepare_room_scenarios.py \
  --librimix_dir ../../data/libriheavymix \
  --rir_dir ../../data/rir \
  --output_dir ./prepared \
  --scenario multi_laptop_room \   # 2-4 speakers, same-room RIR per session
  --n_scenarios 100000

# Train separator (Conv-TasNet based)
python train_separator.py \
  --data_dir ./prepared \
  --model conv_tasnet \
  --n_speakers 4 \
  --causal True \               # must be causal for real-time use
  --latency_target_ms 8 \
  --epochs 100 \
  --output_dir ./checkpoints

# Export ONNX
python export_onnx.py \
  --checkpoint ./checkpoints/best.pt \
  --output ../../pipeline/separation/separator.onnx
```

---

## 7. Core Pipeline Implementation

The pipeline runs in C++ for deterministic latency. ONNX Runtime is used for model inference.

### 7.1 Build the Pipeline

```bash
cd pipeline

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar xf onnxruntime-linux-x64-1.17.1.tgz

# Build
cmake -B build \
  -DONNXRUNTIME_DIR=./onnxruntime-linux-x64-1.17.1 \
  -DWITH_PORTAUDIO=ON \
  -DWITH_OPUS=ON
cmake --build build --parallel
```

### 7.2 Key Pipeline Parameters

```cpp
// pipeline/config.h

// Audio
constexpr int SAMPLE_RATE       = 16000;   // Hz
constexpr int FRAME_SIZE_MS     = 10;      // ms per processing block
constexpr int FRAME_SIZE        = SAMPLE_RATE * FRAME_SIZE_MS / 1000; // 160 samples

// AEC
constexpr float AEC_ERLE_TARGET = 45.0f;  // dB
constexpr int   AEC_LATENCY_MS  = 8;      // max added latency

// Noise
constexpr float NOISE_REDUCTION_DB = 15.0f;
constexpr int   NOISE_LATENCY_MS   = 3;

// PLC
constexpr int PLC_FEC_MAX_MS    = 40;     // Opus RED covers up to 40ms
constexpr int PLC_AI_MAX_MS     = 100;    // LPCNet covers 40-100ms
constexpr int PLC_LATENCY_MS    = 10;     // max added latency

// Encryption
constexpr int FRAME_ENCRYPT_MS  = 10;     // encrypt per frame
constexpr int AES_KEY_BITS      = 256;

// Degradation thresholds
constexpr float CPU_DEGRADE_THRESHOLD = 0.80f;   // disable AI PLC above 80% CPU
constexpr float CPU_FALLBACK_THRESHOLD = 0.95f;  // disable NRES above 95% CPU
```

### 7.3 Latency Budget

| Stage | CPU Budget | GPU Budget |
|-------|-----------|-----------|
| AEC3 (WebRTC) | 3ms | 1ms |
| NRES inference | 4ms | 0.5ms (Maxine) |
| RNNoise / Maxine | 3ms | 0.5ms |
| Source separation | 5ms | 1ms |
| PLC (FEC path) | <1ms | <1ms |
| PLC (LPCNet path) | 5ms | 1ms |
| Encryption | <1ms | <1ms |
| **Total** | **<10ms** | **<3ms** |

---

## 8. Room Coordination Layer

The Room Coordinator lets all laptops in the same room share a lightweight audio reference signal — this is what makes multi-laptop AEC possible.

### 8.1 How It Works

1. Host laptop generates a 6-digit **room code** and starts a local UDP broadcast
2. Other laptops in the same room enter the code → discover each other via LAN broadcast
3. Each laptop shares its **speaker output signal** (what it's playing) with all others at low latency
4. Each laptop's AEC receives the combined room reference = all other laptops' speaker outputs
5. AEC can now cancel echo from any laptop's speaker, not just its own

### 8.2 Implementation

```bash
cd coordinator
pip install asyncio aiohttp cryptography zeroconf
```

```python
# coordinator/discovery.py — LAN peer discovery via mDNS
# coordinator/reference_bus.py — UDP multicast audio reference stream
# coordinator/relay.py — TURN relay fallback when not on same LAN

# Start coordinator (one per room — auto-elected)
python coordinator.py --room-code ABC123 --role host

# Join as participant
python coordinator.py --room-code ABC123 --role peer
```

**Reference signal spec:**
- 8kHz mono, Opus-compressed, ~16kbps
- 20ms jitter buffer
- LAN path: UDP multicast, ~1ms latency
- Relay path: TURN, ~20ms latency

---

## 9. Encryption Layer

### 9.1 1-on-1 Sessions (Signal Protocol)

```bash
pip install cryptography
# Use libsignal-client Python bindings
pip install git+https://github.com/signalapp/libsignal#subdirectory=python
```

```python
# encryption/signal/session.py
# - X3DH key exchange on first connection
# - Double Ratchet for forward secrecy on every message (10ms audio frame)
# - Each encrypted frame: 256-byte fixed size (padded) to prevent length leakage
```

### 9.2 Group Sessions (MLS RFC 9420)

```bash
pip install git+https://github.com/openmls/openmls  # Python bindings
```

```python
# encryption/mls/group.py
# - TreeKEM group key agreement (up to 50 participants)
# - Key update on every participant join/leave
# - Each frame encrypted with AES-256-GCM derived from current epoch key
# - <100ms key agreement latency for 10-person group
```

### 9.3 Zero-Knowledge Relay

The relay server forwards encrypted frames but **never holds decryption keys**. Keys are exchanged peer-to-peer via the Signal/MLS handshake. The relay only sees ciphertext.

---

## 10. NVIDIA Maxine Integration

```cpp
// maxine/maxine_pipeline.cpp

#include "nvAudioEffects.h"

class MaxinePipeline {
    NvAFX_Handle aec_handle;
    NvAFX_Handle denoiser_handle;
    NvAFX_Handle superres_handle;

public:
    bool init(const std::string& sdk_path) {
        // AEC
        NvAFX_CreateEffect(NVAFX_EFFECT_AEC, &aec_handle);
        NvAFX_SetU32(aec_handle, NVAFX_PARAM_AEC_ENABLE_VAD, 1);

        // Denoiser
        NvAFX_CreateEffect(NVAFX_EFFECT_DENOISER, &denoiser_handle);
        NvAFX_SetF32(denoiser_handle, NVAFX_PARAM_DENOISER_INTENSITY, 1.0f);

        // Super Resolution (8kHz → 16kHz)
        NvAFX_CreateEffect(NVAFX_EFFECT_SUPER_RESOLUTION, &superres_handle);

        return NvAFX_Load(aec_handle) == NVAFX_STATUS_SUCCESS;
    }

    float* process(float* mic_in, float* ref_in, int n_samples) {
        NvAFX_Run(aec_handle, &mic_in, &ref_in, &aec_out, n_samples);
        NvAFX_Run(denoiser_handle, &aec_out, &denoised_out, n_samples);
        NvAFX_Run(superres_handle, &denoised_out, &final_out, n_samples);
        return final_out;
    }
};
```

**Fallback logic:** if `NvAFX_Load` fails (no GPU), pipeline constructor automatically instantiates the CPU path (NRES + RNNoise). Re-checked every 30 seconds.

---

## 11. Desktop App & Virtual Audio Device

### 11.1 Virtual Audio Device

The virtual audio device makes RoomSync appear as a microphone to Zoom, Teams, and Meet — no plugin needed on the remote end.

**macOS (CoreAudio HAL plugin):**
```bash
cd virtual_device/macos
# Based on BlackHole open-source driver
cmake -B build && cmake --build build
sudo cp RoomSync.driver /Library/Audio/Plug-Ins/HAL/
sudo launchctl kickstart -k system/com.apple.audio.coreaudiod
```

**Windows (WASAPI virtual device):**
```bash
cd virtual_device/windows
# Based on VB-Audio Virtual Cable pattern
msbuild RoomSyncVirtualDevice.sln /p:Configuration=Release
# Install the driver (requires admin + code signing for production)
pnputil /add-driver RoomSync.inf /install
```

### 11.2 Desktop App (Tauri + React)

```bash
# Install Tauri CLI
cargo install tauri-cli

cd app
npm install

# Development
npm run tauri dev

# Build for distribution
npm run tauri build
```

**App responsibilities:**
- Room code display and entry
- Connection status (how many laptops in room)
- Audio quality indicators (latency, packet loss %, echo level)
- Settings: GPU toggle, tier selection, encryption status
- Tray icon (always-on-top indicator when active)

---

## 12. Testing & Benchmarking

### 12.1 Latency Measurement

```bash
cd tests/latency
# Plays a click through the pipeline, measures round-trip
python measure_latency.py --iterations 1000 --device RoomSync
# Target: p99 < 10ms CPU, < 3ms GPU
```

### 12.2 ERLE (Echo Suppression Quality)

```bash
cd tests/erle
# Uses AEC Challenge test set
python measure_erle.py \
  --test_dir ../../data/AEC-Challenge/test \
  --pipeline ./build/roomsync_pipeline
# Target: >45dB ERLE across all test conditions
```

### 12.3 MOS (Perceptual Quality)

```bash
cd tests/mos
# Uses DNSMOS P.835 (Microsoft's non-intrusive MOS predictor)
pip install onnxruntime
python measure_mos.py \
  --audio_dir ./samples \
  --model dnsmos_p835.onnx
# Target: MOS >4.0 on SIG (speech quality) axis
```

### 12.4 Packet Loss Recovery

```bash
cd tests/plc
# Simulates Wi-Fi packet loss patterns, measures recovery quality
python test_plc.py \
  --loss_rate 0.03 \         # 3% packet loss
  --burst_ms 100 \           # 100ms burst loss events
  --audio ./samples/speech.wav
# Target: PESQ >3.5 after reconstruction
```

### 12.5 CPU Usage

```bash
cd tests/
# Measures CPU usage with 1, 2, 4 active speakers in room
python measure_cpu.py --speakers 4 --duration 60
# Target: <20% CPU on Intel i7-1265U with 4 active speakers
```

---

## 13. Week-by-Week Milestones

| Week | Owner | Deliverable | Done When |
|------|-------|------------|-----------|
| **1** | AI Eng 1 | AEC pipeline: WebRTC AEC3 running in C++ pipeline | ERLE >40dB on single-laptop test |
| **2** | AI Eng 1 | NRES model trained on AEC dataset, integrated | ERLE >45dB on double-talk test |
| **3** | AI Eng 2 | RNNoise trained on DNS-5, integrated | -15dB noise reduction at <3ms |
| **4** | AI Eng 2 | LPCNet trained on PLC dataset | PESQ >3.5, <5ms inference |
| **5** | Full Stack | Room Coordinator: room code, LAN discovery, reference bus | 2 laptops sharing audio reference, <5ms LAN |
| **6** | Full Stack | End-to-end pipeline: AEC→Noise→PLC under 10ms | p99 latency <10ms, 4-speaker test |
| **7** | AI Eng 1 | Signal Protocol (1:1) + MLS (group) E2E encryption | Encrypted frames, server cannot decrypt |
| **8** | AI Eng 2 | Source separator trained on LibriheavyMix | Cross-laptop voice suppressed >20dB |
| **9** | Full Stack | NVIDIA Maxine GPU path, CPU<→GPU fallback | <3ms GPU, fallback in <100ms |
| **10** | Product | Virtual audio device (macOS + Windows), Tauri app shell | Zoom selects RoomSync as mic input |
| **11** | Product + AI | Full integration test: 4 laptops, same room, Zoom call | MOS >4.0, no echo, no artifacts |
| **12** | All | Closed beta: 10 design partners, telemetry, packaging | NPS >50, p99 latency <10ms in production |

---

## 14. Success Metrics

| Metric | Target | Measured By |
|--------|--------|------------|
| Echo suppression (ERLE) | >45 dB | AEC Challenge test set |
| Noise reduction | -15 dB | DNS-5 test set |
| Packet loss recovery | Up to 100ms reconstructed | ICASSP 2024 PLC blind test |
| Speech quality (MOS) | >4.0 | DNSMOS P.835 |
| Pipeline latency (CPU) | <10ms p99 | Loopback measurement, 1000 iterations |
| Pipeline latency (GPU) | <3ms p99 | `nvprof` + loopback |
| CPU usage (4 speakers) | <20% | i7-1265U, 60s sustained |
| RAM footprint | <150MB | Process monitor, all models loaded |
| Beta NPS | >50 | 10 design partner survey |

---

## Quick Start (Dev Mode)

```bash
# 1. Clone and set up
git clone https://github.com/yourorg/roomsync
cd roomsync
./scripts/setup.sh              # installs all dependencies

# 2. Download minimal dataset for dev (AEC only, 10GB)
cd training/datasets
./download_aec.sh --dev         # dev split only

# 3. Train NRES (fast, 1hr on RTX 3060)
cd ../nres
python train.py --epochs 10 --fast-dev

# 4. Build pipeline
cd ../../
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build

# 5. Run pipeline test
./build/roomsync_test --input ./tests/samples/double_talk.wav

# 6. Launch app
cd app && npm run tauri dev
```

---

*Generated by RoomSync Autodiscover Research Pipeline · Score 78/100 · May 2026*

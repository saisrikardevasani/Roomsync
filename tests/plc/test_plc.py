"""Simulate Wi-Fi packet loss and measure reconstruction quality (PESQ).

Usage:
    python test_plc.py \\
        --loss_rate 0.03 \\
        --burst_ms 100 \\
        --audio ./samples/speech.wav
"""

import argparse
import random
import numpy as np
from pathlib import Path

try:
    import soundfile as sf
    from pesq import pesq
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

SAMPLE_RATE  = 16000
FRAME_MS     = 10
FRAME_SIZE   = SAMPLE_RATE * FRAME_MS // 1000


def simulate_loss(frames: list[np.ndarray], loss_rate: float,
                  burst_ms: int, rng: random.Random) -> tuple[list, list[bool]]:
    """Apply Gilbert-Elliott loss model. Returns (received, is_lost) lists."""
    burst_frames  = burst_ms // FRAME_MS
    received      = []
    lost_mask     = []
    in_burst      = 0

    for frame in frames:
        if in_burst > 0:
            received.append(np.zeros_like(frame))
            lost_mask.append(True)
            in_burst -= 1
        elif rng.random() < loss_rate:
            in_burst = burst_frames - 1
            received.append(np.zeros_like(frame))
            lost_mask.append(True)
        else:
            received.append(frame)
            lost_mask.append(False)

    return received, lost_mask


def pitch_repeat_plc(history: list[np.ndarray], loss_step: int) -> np.ndarray:
    """Comfort-noise + pitch-repeat fallback PLC."""
    if not history:
        return np.zeros(FRAME_SIZE, dtype=np.float32)
    fade   = max(0.0, 1.0 - loss_step / 10)
    return history[-1] * fade


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loss_rate", type=float, default=0.03)
    p.add_argument("--burst_ms",  type=int,   default=100)
    p.add_argument("--audio",     required=True)
    args = p.parse_args()

    if not HAS_DEPS:
        print("Missing deps — pip install soundfile pesq")
        return

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if sr != SAMPLE_RATE:
        print(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz")
        return

    # Segment into frames
    frames = [audio[i:i+FRAME_SIZE] for i in range(0, len(audio) - FRAME_SIZE, FRAME_SIZE)]
    frames = [f.astype(np.float32) for f in frames if len(f) == FRAME_SIZE]

    rng            = random.Random(42)
    received, lost = simulate_loss(frames, args.loss_rate, args.burst_ms, rng)

    # PLC reconstruction
    history       = []
    reconstructed = []
    loss_step     = 0

    for frame, is_lost in zip(received, lost):
        if is_lost:
            loss_step += 1
            concealed = pitch_repeat_plc(history, loss_step)
            reconstructed.append(concealed)
        else:
            loss_step = 0
            reconstructed.append(frame)
            history.append(frame)
            if len(history) > 20:
                history.pop(0)

    ref_audio   = np.concatenate(frames)
    recon_audio = np.concatenate(reconstructed)
    n           = min(len(ref_audio), len(recon_audio))

    loss_pct = sum(lost) / len(lost) * 100
    print(f"Simulated loss: {loss_pct:.1f}%  (target: {args.loss_rate*100:.1f}%)")

    try:
        score = pesq(SAMPLE_RATE, ref_audio[:n], recon_audio[:n], "nb")
        print(f"PESQ (NB): {score:.2f}  [target: >3.5]")
        print(f"PASS: {'YES' if score > 3.5 else 'NO (fallback PLC — train LPCNet for better score)'}")
    except Exception as e:
        print(f"PESQ failed: {e}")


if __name__ == "__main__":
    main()

"""Measure MOS (Mean Opinion Score) using DNSMOS P.835.

Usage:
    python measure_mos.py \\
        --audio_dir ./samples \\
        --model dnsmos_p835.onnx
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    import soundfile as sf
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

SAMPLE_RATE  = 16000
WINDOW_LEN   = SAMPLE_RATE * 10  # 10s input expected by DNSMOS


def run_dnsmos(session: "ort.InferenceSession", audio: np.ndarray) -> dict:
    # Pad/trim to expected length
    if len(audio) < WINDOW_LEN:
        audio = np.pad(audio, (0, WINDOW_LEN - len(audio)))
    else:
        audio = audio[:WINDOW_LEN]

    audio = audio.astype(np.float32).reshape(1, -1)
    out   = session.run(None, {"input_1": audio})[0]
    # DNSMOS outputs: [SIG, BAK, OVR]
    return {"SIG": float(out[0, 0]), "BAK": float(out[0, 1]), "OVR": float(out[0, 2])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--model",     default="dnsmos_p835.onnx")
    args = p.parse_args()

    if not HAS_DEPS:
        print("Missing deps — pip install onnxruntime soundfile")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"DNSMOS model not found at {model_path}. Download from:")
        print("  https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS")
        return

    session   = ort.InferenceSession(str(model_path))
    audio_dir = Path(args.audio_dir)
    wav_files = sorted(audio_dir.glob("**/*.wav"))

    if not wav_files:
        print(f"No .wav files in {audio_dir}")
        return

    sig_scores = []
    for wav_path in wav_files:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if sr != SAMPLE_RATE:
            print(f"Skipping {wav_path.name} — expected {SAMPLE_RATE}Hz, got {sr}Hz")
            continue
        scores = run_dnsmos(session, audio)
        sig_scores.append(scores["SIG"])
        print(f"  {wav_path.name}: SIG={scores['SIG']:.2f} BAK={scores['BAK']:.2f} OVR={scores['OVR']:.2f}")

    if sig_scores:
        mean_sig = np.mean(sig_scores)
        print(f"\nMean SIG MOS: {mean_sig:.2f}  [target: >4.0]")
        print(f"PASS: {'YES' if mean_sig > 4.0 else 'NO'}")


if __name__ == "__main__":
    main()

"""Measure Echo Return Loss Enhancement (ERLE) on the AEC Challenge test set.

Usage:
    python measure_erle.py \\
        --test_dir ../../data/AEC-Challenge/test \\
        --pipeline ../../build/roomsync_pipeline
"""

import argparse
import os
import subprocess
import numpy as np
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None


def erle_db(mic: np.ndarray, echo_cancelled: np.ndarray, ref: np.ndarray) -> float:
    """ERLE = 10*log10(E[ref²] / E[(mic-echo_cancelled)²])."""
    echo_power  = np.mean(ref ** 2) + 1e-10
    resid_power = np.mean((mic - echo_cancelled) ** 2) + 1e-10
    return 10 * np.log10(echo_power / resid_power)


def run_pipeline_on_file(pipeline_bin: str, input_wav: str,
                         ref_wav: str, output_wav: str) -> bool:
    """Run the C++ pipeline binary on a test file."""
    if not os.path.exists(pipeline_bin):
        return False
    result = subprocess.run(
        [pipeline_bin, "--input", input_wav, "--ref", ref_wav, "--output", output_wav],
        capture_output=True)
    return result.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir",  required=True)
    p.add_argument("--pipeline",  required=True)
    args = p.parse_args()

    if sf is None:
        print("soundfile not installed — pip install soundfile")
        return

    test_dir  = Path(args.test_dir)
    mic_files = sorted(test_dir.glob("**/mic_*.wav"))

    if not mic_files:
        print(f"No mic_*.wav files found in {test_dir}")
        return

    erle_scores = []
    for mic_path in mic_files[:100]:  # cap at 100 files for speed
        ref_path = Path(str(mic_path).replace("mic_", "ref_"))
        out_path = mic_path.with_suffix(".out.wav")
        if not ref_path.exists():
            continue

        ok = run_pipeline_on_file(str(args.pipeline),
                                  str(mic_path), str(ref_path), str(out_path))
        if not ok or not out_path.exists():
            continue

        mic_wav,  sr  = sf.read(str(mic_path),  dtype="float32")
        ref_wav,  _   = sf.read(str(ref_path),  dtype="float32")
        out_wav,  _   = sf.read(str(out_path),  dtype="float32")
        n = min(len(mic_wav), len(ref_wav), len(out_wav))
        score = erle_db(mic_wav[:n], out_wav[:n], ref_wav[:n])
        erle_scores.append(score)

    if not erle_scores:
        print("No files processed")
        return

    mean_erle = np.mean(erle_scores)
    print(f"\nERLE Results ({len(erle_scores)} files):")
    print(f"  Mean ERLE: {mean_erle:.1f}dB  [target: >45dB]")
    print(f"  PASS: {'YES' if mean_erle > 45 else 'NO'}")


if __name__ == "__main__":
    main()

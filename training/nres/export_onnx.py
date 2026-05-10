"""Export trained NRES checkpoint to ONNX for C++ inference."""
import argparse
import torch
from train import DCCRN
from pathlib import Path


def export(args: argparse.Namespace) -> None:
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = DCCRN()
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dummy input: (batch=1, freq_bins=257, frames=32)
    dummy_r = torch.zeros(1, 257, 32)
    dummy_i = torch.zeros(1, 257, 32)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy_r, dummy_i), args.output,
        input_names=["input_real", "input_imag"],
        output_names=["output_real", "output_imag"],
        opset_version=14,
        dynamic_axes={"input_real": {2: "frames"}, "input_imag": {2: "frames"},
                      "output_real": {2: "frames"}, "output_imag": {2: "frames"}},
    )
    print(f"Exported NRES to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     required=True)
    export(p.parse_args())

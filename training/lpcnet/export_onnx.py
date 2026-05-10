"""Export LPCNet checkpoint to ONNX for C++ inference."""
import argparse
import torch
from train_lpcnet import LPCNetPLC, HISTORY_FRAMES, FRAME_SAMPLES
from pathlib import Path


def export(args: argparse.Namespace) -> None:
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = LPCNetPLC(model_size="small")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_ctx = torch.zeros(1, HISTORY_FRAMES, FRAME_SAMPLES)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy_ctx, args.output,
        input_names=["context"],
        output_names=["concealed"],
        opset_version=14,
        dynamic_axes={"context": {0: "batch"}},
    )
    print(f"Exported LPCNet PLC to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     required=True)
    export(p.parse_args())

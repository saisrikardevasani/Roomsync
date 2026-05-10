"""Export separator checkpoint to ONNX."""
import argparse, torch
from train_separator import ConvTasNet
from pathlib import Path


def export(args):
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = ConvTasNet()
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 1, 16000)  # 1s at 16kHz
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"], output_names=["output_0"],
        opset_version=14,
        dynamic_axes={"input": {2: "time"}, "output_0": {2: "time"}},
    )
    print(f"Exported separator to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     required=True)
    export(p.parse_args())

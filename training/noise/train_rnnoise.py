"""RNNoise-fork training — 3-layer GRU trained on DNS-5.

Targets: -15dB noise reduction, SI-SDR >18dB, inference <3ms ARM Cortex-A76.

Usage:
    python train_rnnoise.py \\
        --data_dir ./mixed \\
        --gru_layers 3 \\
        --hidden_size 96 \\
        --epochs 150 \\
        --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

FRAME_SAMPLES = 480   # 30ms at 16kHz (RNNoise convention)
N_BANDS       = 22    # Bark-scale frequency bands


class RNNoiseGRU(nn.Module):
    """3-layer GRU noise suppressor that outputs per-band gain masks."""

    def __init__(self, gru_layers: int = 3, hidden_size: int = 96) -> None:
        super().__init__()
        self.feature_fc  = nn.Linear(N_BANDS, 64)
        self.gru         = nn.GRU(64, hidden_size, num_layers=gru_layers,
                                   batch_first=True)
        self.mask_fc     = nn.Sequential(
            nn.Linear(hidden_size, N_BANDS),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, N_BANDS)
        x   = torch.relu(self.feature_fc(features))
        gru_out, _ = self.gru(x)
        return self.mask_fc(gru_out)   # (B, T, N_BANDS) gain mask ∈ [0,1]


class DNSDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.files = sorted(Path(data_dir).glob("*.pt"))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx])
        return d["noisy_features"], d["clean_features"]


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RNNoiseGRU(args.gru_layers, args.hidden_size).to(device)
    opt    = optim.Adam(model.parameters(), lr=3e-4)

    dataset    = DNSDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            mask = model(noisy)
            loss = nn.MSELoss()(mask * noisy, clean)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / max(len(dataloader), 1)
        print(f"Epoch {epoch}/{args.epochs}  MSE={avg:.5f}")
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict()}, ckpt_dir / "best.pt")

    print(f"Done. Best MSE={best:.5f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--gru_layers",  type=int, default=3)
    p.add_argument("--hidden_size", type=int, default=96)
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--output_dir",  default="./checkpoints")
    train(p.parse_args())


if __name__ == "__main__":
    main()

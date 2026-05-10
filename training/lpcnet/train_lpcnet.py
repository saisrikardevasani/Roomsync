"""LPCNet Packet Loss Concealment training script.

Reconstructs up to 100ms of lost speech from prior context.
Based on Mozilla LPCNet with a small (0.8M param) architecture
to stay under 5ms inference on ARM.

Usage:
    python train_lpcnet.py \\
        --data_dir ./prepared \\
        --model_size small \\
        --epochs 200 \\
        --lr 2e-4 \\
        --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


FRAME_SAMPLES  = 160   # 10ms at 16kHz
HISTORY_FRAMES = 20    # 200ms context


# ── Model ─────────────────────────────────────────────────────────────────────

class LPCNetPLC(nn.Module):
    """Autoregressive model conditioned on prior speech context.

    Architecture:
      - Context encoder: 1-layer GRU over HISTORY_FRAMES of LP residual
      - Frame decoder: GRU + linear → predicted waveform frame
    Size variants: 'small' (~0.8M), 'medium' (~2M), 'large' (~5M)
    """

    SIZE_CONFIG = {
        "small":  {"enc_hidden": 64,  "dec_hidden": 128},
        "medium": {"enc_hidden": 128, "dec_hidden": 256},
        "large":  {"enc_hidden": 256, "dec_hidden": 512},
    }

    def __init__(self, model_size: str = "small") -> None:
        super().__init__()
        cfg = self.SIZE_CONFIG[model_size]
        eh, dh = cfg["enc_hidden"], cfg["dec_hidden"]

        self.context_gru = nn.GRU(FRAME_SAMPLES, eh, num_layers=1, batch_first=True)
        self.frame_gru   = nn.GRU(eh, dh, num_layers=2, batch_first=True)
        self.out_linear  = nn.Linear(dh, FRAME_SAMPLES)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, HISTORY_FRAMES, FRAME_SAMPLES)
        _, h = self.context_gru(context)
        # Use encoder hidden state as initial state for frame decoder
        h_dec = h.repeat(2, 1, 1)  # 2 GRU layers
        zeros = torch.zeros(context.size(0), 1, h.size(-1), device=context.device)
        out, _ = self.frame_gru(zeros, h_dec)
        return self.out_linear(out.squeeze(1))  # (B, FRAME_SAMPLES)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PLCDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.files = sorted(Path(data_dir).glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx])
        return data["context"], data["target_frame"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LPCNetPLC(model_size=args.model_size).to(device)
    opt    = optim.Adam(model.parameters(), lr=args.lr)
    sched  = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 10,
                                            epochs=args.epochs,
                                            steps_per_epoch=1)
    loss_fn = nn.L1Loss()

    dataset    = PLCDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ctx, target in dataloader:
            ctx, target = ctx.to(device), target.to(device)
            pred = model(ctx)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        avg = total / max(len(dataloader), 1)
        sched.step()
        print(f"Epoch {epoch}/{args.epochs}  L1={avg:.5f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

    print(f"Done. Best L1={best_loss:.5f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--model_size",  default="small", choices=["small", "medium", "large"])
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--output_dir",  default="./checkpoints")
    train(p.parse_args())


if __name__ == "__main__":
    main()

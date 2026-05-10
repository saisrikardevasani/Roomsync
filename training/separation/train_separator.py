"""Conv-TasNet source separator training — causal mode for real-time use.

Targets: cross-laptop voice suppressed >20dB, latency <8ms.

Usage:
    python train_separator.py \\
        --data_dir ./prepared \\
        --model conv_tasnet \\
        --n_speakers 4 \\
        --causal True \\
        --epochs 100 \\
        --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ── Conv-TasNet (causal) ──────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel: int, dilation: int,
                 causal: bool) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation if causal else (kernel - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation,
                      padding=pad, groups=hidden),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, in_ch, 1),
        )
        self.causal  = causal
        self.trim    = (kernel - 1) * dilation if causal else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.causal and self.trim > 0:
            out = out[..., :-self.trim]
        return x + out


class ConvTasNet(nn.Module):
    """Causal Conv-TasNet for real-time source separation."""

    def __init__(self, n_speakers: int = 4, causal: bool = True,
                 n_feats: int = 256, n_hidden: int = 512,
                 n_blocks: int = 8, n_repeats: int = 3,
                 kernel_size: int = 16, enc_dim: int = 512) -> None:
        super().__init__()
        self.n_speakers = n_speakers
        self.encoder    = nn.Conv1d(1, enc_dim, kernel_size, stride=kernel_size // 2,
                                    bias=False)
        self.encoder_bn = nn.GroupNorm(1, enc_dim)

        # TCN blocks
        tcn_blocks = []
        for r in range(n_repeats):
            for b in range(n_blocks):
                dil = 2 ** b
                tcn_blocks.append(
                    DepthwiseSeparableConv(enc_dim, n_hidden, 3, dil, causal))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.mask_head = nn.Sequential(
            nn.Conv1d(enc_dim, enc_dim * n_speakers, 1),
            nn.ReLU(),
        )
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size,
                                          stride=kernel_size // 2, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: (B, 1, T)
        enc  = torch.relu(self.encoder_bn(self.encoder(mixture)))  # (B, E, L)
        feat = self.tcn(enc)
        masks = self.mask_head(feat)                                # (B, E*S, L)
        B, _, L = feat.shape
        masks = masks.view(B, self.n_speakers, -1, L)              # (B, S, E, L)
        masks = torch.softmax(masks, dim=1)

        # Return first source (near-end speaker)
        src_enc = feat.unsqueeze(1) * masks[:, 0:1]                # (B, 1, E, L)
        return self.decoder(src_enc.squeeze(1))                     # (B, 1, T)


# ── Loss ──────────────────────────────────────────────────────────────────────

def si_snr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred - pred.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    s_tgt  = (pred * target).sum(-1, keepdim=True) / ((target ** 2).sum(-1, keepdim=True) + 1e-8) * target
    e_noise = pred - s_tgt
    return -10 * torch.log10((s_tgt**2).sum(-1) / ((e_noise**2).sum(-1) + 1e-8) + 1e-8).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SeparationDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.files = sorted(Path(data_dir).glob("*.pt"))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        d = torch.load(self.files[idx])
        return d["mixture"], d["near_end"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ConvTasNet(n_speakers=args.n_speakers, causal=args.causal).to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)

    dataset    = SeparationDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for mix, near in dataloader:
            mix, near = mix.to(device), near.to(device)
            pred = model(mix.unsqueeze(1))
            loss = si_snr(pred.squeeze(1), near)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        avg = total / max(len(dataloader), 1)
        print(f"Epoch {epoch}/{args.epochs}  SI-SNR-loss={avg:.3f}")
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict()}, ckpt_dir / "best.pt")

    print(f"Done. Best SI-SNR-loss={best:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--model",       default="conv_tasnet")
    p.add_argument("--n_speakers",  type=int, default=4)
    p.add_argument("--causal",      type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--latency_target_ms", type=int, default=8)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--output_dir",  default="./checkpoints")
    train(p.parse_args())


if __name__ == "__main__":
    main()

"""NRES (Neural Residual Echo Suppressor) training script.

Architecture: 4-layer DCCRN (Deep Complex Convolutional Recurrent Network)
  - Input:  complex spectrogram from mic + AEC3 residual (257 bins, 16kHz)
  - Layers: 4 encoder (complex conv) + 2 LSTM + 4 decoder
  - Params: ~1.2M
  - Output: clean speech mask applied to residual

Usage:
    python train.py \\
        --data_dir ./prepared \\
        --model dccrn \\
        --layers 4 \\
        --batch_size 16 \\
        --epochs 100 \\
        --lr 1e-3 \\
        --loss sisnr+mse \\
        --checkpoint_dir ./checkpoints \\
        --wandb_project roomsync-nres
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio


# ── Model ─────────────────────────────────────────────────────────────────────

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: tuple, stride: tuple = (1, 1),
                 padding: tuple = (0, 0)) -> None:
        super().__init__()
        self.real_conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.imag_conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        return (self.real_conv(real) - self.imag_conv(imag),
                self.real_conv(imag) + self.imag_conv(real))


class DCCRN(nn.Module):
    """Deep Complex Convolutional Recurrent Network for residual echo suppression."""

    def __init__(self, n_layers: int = 4, fft_size: int = 512,
                 hidden_size: int = 256) -> None:
        super().__init__()
        self.fft_size    = fft_size
        self.n_fft_bins  = fft_size // 2 + 1  # 257
        self.n_layers    = n_layers

        # Encoder: complex conv layers
        self.encoders = nn.ModuleList([
            ComplexConv2d(1 if i == 0 else 2**i,
                          2**(i + 1),
                          (3, 2), (1, 2), (1, 0))
            for i in range(n_layers)
        ])
        self.encoder_bn = nn.ModuleList([nn.BatchNorm2d(2**(i+1)) for i in range(n_layers)])

        # Bottleneck: 2 LSTM layers
        lstm_input = 2**n_layers * (self.n_fft_bins // (2**n_layers))
        self.lstm = nn.LSTM(lstm_input * 2, hidden_size, num_layers=2,
                            batch_first=True, bidirectional=False)
        self.lstm_fc = nn.Linear(hidden_size, lstm_input * 2)

        # Decoder: complex transposed conv layers
        self.decoders = nn.ModuleList([
            nn.ConvTranspose2d(2**(n_layers - i) * 2,
                               2**(n_layers - i - 1) if i < n_layers - 1 else 2,
                               (3, 2), (1, 2), (1, 0))
            for i in range(n_layers)
        ])

        self.mask_act = nn.Tanh()

    def forward(self, noisy_real: torch.Tensor, noisy_imag: torch.Tensor):
        B, F, T = noisy_real.shape  # (batch, freq_bins, frames)

        # Encode
        er, ei = noisy_real.unsqueeze(1), noisy_imag.unsqueeze(1)
        skip_real, skip_imag = [], []
        for enc, bn in zip(self.encoders, self.encoder_bn):
            er, ei = enc(er, ei)
            er = bn(er)
            skip_real.append(er)
            skip_imag.append(ei)

        # LSTM bottleneck
        B2, C2, F2, T2 = er.shape
        bottleneck = torch.cat([er, ei], dim=1).permute(0, 3, 1, 2)
        bottleneck = bottleneck.reshape(B2, T2, -1)
        lstm_out, _ = self.lstm(bottleneck)
        lstm_out     = self.lstm_fc(lstm_out)
        lstm_out     = lstm_out.reshape(B2, T2, C2 * 2, F2).permute(0, 2, 3, 1)
        er = lstm_out[:, :C2]
        ei = lstm_out[:, C2:]

        # Decode with skip connections
        for dec in self.decoders:
            er = dec(torch.cat([er, skip_real.pop()], dim=1))
            ei = dec(torch.cat([ei, skip_imag.pop()], dim=1))

        # Complex mask
        mask_r = self.mask_act(er.squeeze(1))
        mask_i = self.mask_act(ei.squeeze(1))
        out_r  = noisy_real * mask_r - noisy_imag * mask_i
        out_i  = noisy_real * mask_i + noisy_imag * mask_r
        return out_r, out_i


# ── Loss ──────────────────────────────────────────────────────────────────────

def si_snr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SNR loss (higher = better, so we negate)."""
    pred   = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot    = (pred * target).sum(dim=-1, keepdim=True)
    t_norm = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
    s_target = dot / t_norm * target
    e_noise  = pred - s_target
    si_snr   = 10 * torch.log10((s_target ** 2).sum(-1) / ((e_noise ** 2).sum(-1) + 1e-8) + 1e-8)
    return -si_snr.mean()


def combined_loss(pred_r, pred_i, tgt_r, tgt_i):
    pred_wav  = torch.istft(torch.stack([pred_r, pred_i], dim=-1), n_fft=512, hop_length=256)
    tgt_wav   = torch.istft(torch.stack([tgt_r,  tgt_i],  dim=-1), n_fft=512, hop_length=256)
    return si_snr_loss(pred_wav, tgt_wav) + nn.MSELoss()(pred_r, tgt_r) + nn.MSELoss()(pred_i, tgt_i)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AECDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 16000, segment_s: float = 2.0) -> None:
        self.files       = sorted(Path(data_dir).glob("*.pt"))
        self.sample_rate = sample_rate
        self.seg_len     = int(sample_rate * segment_s)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx])
        return data["mic_residual"], data["clean_speech"]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = DCCRN(n_layers=args.layers).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dataset    = AECDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        use_wandb = True
    except ImportError:
        use_wandb = False

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for mic, clean in dataloader:
            mic, clean = mic.to(device), clean.to(device)

            # STFT
            mic_stft   = torch.stft(mic.squeeze(1),   n_fft=512, hop_length=256, return_complex=False)
            clean_stft = torch.stft(clean.squeeze(1), n_fft=512, hop_length=256, return_complex=False)

            mic_r, mic_i     = mic_stft[..., 0].permute(0, 2, 1), mic_stft[..., 1].permute(0, 2, 1)
            clean_r, clean_i = clean_stft[..., 0].permute(0, 2, 1), clean_stft[..., 1].permute(0, 2, 1)

            pred_r, pred_i = model(mic_r, mic_i)
            loss           = combined_loss(pred_r, pred_i, clean_r, clean_i)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        sched.step()
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")

        if use_wandb:
            wandb.log({"loss": avg_loss, "epoch": epoch})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": avg_loss},
                       ckpt_dir / "best.pt")

    print(f"Training complete. Best loss: {best_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        required=True)
    parser.add_argument("--model",           default="dccrn")
    parser.add_argument("--layers",          type=int, default=4)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--loss",            default="sisnr+mse")
    parser.add_argument("--checkpoint_dir",  default="./checkpoints")
    parser.add_argument("--wandb_project",   default="roomsync-nres")
    parser.add_argument("--fast-dev",        action="store_true")
    args = parser.parse_args()

    if args.fast_dev:
        args.epochs = 2
        args.batch_size = 4

    train(args)


if __name__ == "__main__":
    main()

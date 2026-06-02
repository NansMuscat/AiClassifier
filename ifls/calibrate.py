"""
Post-hoc confidence calibration via temperature scaling.

A single scalar T is learned on a held-out calibration set:
    calibrated_probs = softmax(logits / T)

T > 1 → model was overconfident (typical with fine-tuned transformers).
T < 1 → model was underconfident.

The scaler is saved as a plain .pt file and loaded at inference time.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """
        Fit temperature on held-out (logits, labels) via L-BFGS.
        Both tensors should be on CPU.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        nll_before = F.cross_entropy(logits, labels).item()
        nll_after  = F.cross_entropy(logits / self.temperature, labels).item()
        print(
            f"[calibration] T={self.temperature.item():.4f}  "
            f"NLL {nll_before:.4f} → {nll_after:.4f}"
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        scaler = cls()
        scaler.load_state_dict(torch.load(path, map_location="cpu"))
        scaler.eval()
        return scaler

from __future__ import annotations

from pathlib import Path

import torch

from video_recsys.model import ModelConfig, TransformerRanker
from video_recsys.synthetic import synthetic_batch


def train_demo(steps: int = 40, output: Path | None = None) -> list[float]:
    torch.manual_seed(42)
    config = ModelConfig()
    model = TransformerRanker(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses = []
    model.train()
    for step in range(steps):
        batch, watch, purchase = synthetic_batch(config=config, seed=step)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = model.multitask_loss(outputs, watch, purchase)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output)
    return losses


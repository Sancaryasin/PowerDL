from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from contextlib import nullcontext


@dataclass
class TorchRunConfig:
    device: str = "cuda"
    amp: bool = True


def train_one_epoch(model, loader, optimizer, loss_fn, rec, epoch: int, cfg: TorchRunConfig):
    model.train()
    rec.mark("epoch_start", epoch=int(epoch))

    use_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    amp_enabled = bool(cfg.amp and use_cuda)

    # New AMP API (PyTorch 2.x)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if use_cuda else None
    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=amp_enabled) if use_cuda else nullcontext()
    )

    for bidx, (x, y) in enumerate(loader):
        rec.mark("batch_start", batch=int(bidx))

        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(x)
            loss = loss_fn(logits, y)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = int(x.shape[0])
        rec.mark("batch_end", batch=int(bidx), batch_size=bs, loss=float(loss.detach().cpu().item()))

    rec.mark("epoch_end", epoch=int(epoch))


@torch.no_grad()
def inference(model, loader, rec, cfg: TorchRunConfig, max_batches: Optional[int] = None):
    model.eval()
    rec.mark("infer_start")
    n = 0
    for i, (x, _y) in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        x = x.to(cfg.device, non_blocking=True)
        _ = model(x)
        n += int(x.shape[0])
    rec.mark("infer_end", n_samples=n)
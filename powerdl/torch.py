from __future__ import annotations
import time
from dataclasses import dataclass

import torch

@dataclass
class TorchRunConfig:
    device: str = "cuda"
    amp: bool = True

def train_one_epoch(model, loader, optimizer, loss_fn, rec, epoch: int, cfg: TorchRunConfig):
    model.train()
    rec.mark("epoch_start", epoch=int(epoch))

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    for bidx, (x, y) in enumerate(loader):
        rec.mark("batch_start", batch=int(bidx))

        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(x.shape[0])
        rec.mark("batch_end", batch=int(bidx), batch_size=bs, loss=float(loss.detach().cpu().item()))

    rec.mark("epoch_end", epoch=int(epoch))

@torch.no_grad()
def inference(model, loader, rec, cfg: TorchRunConfig, max_batches: int | None = None):
    model.eval()
    rec.mark("infer_start")
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(cfg.device, non_blocking=True)
        _ = model(x)
        n += int(x.shape[0])
    rec.mark("infer_end", n_samples=n)

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.data import load_config, PriceDataset
from model.model import Architecture3


def train(cfg: SimpleNamespace) -> Architecture3:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_set = PriceDataset(
        csv_path=cfg.data_path,
        T=cfg.T,
        price_cols_prefix=cfg.price_cols_prefix,
        prob_cols_prefix=cfg.prob_cols_prefix,
    )
    loader = DataLoader(train_set, batch_size=cfg.batch_size,
                        shuffle=True, drop_last=True)

    model = Architecture3(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[arch3] parameters: {n_params/1e6:.2f}M  device={device}  T+1={cfg.T+1}  d_z={cfg.d_z}  examples={len(train_set)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.n_epochs * steps_per_epoch
    )

    model.train()
    global_step = 0
    L_total = cfg.T + 1

    for epoch in range(cfg.n_epochs):
        for prices_batch, true_probs_batch in loader:
            prices_batch = prices_batch.to(device)               # (B, T+1)
            true_probs_batch = true_probs_batch.to(device)       # (B, T+1)
            B = prices_batch.shape[0]

            
            input_seq = torch.empty(B, 0, 2, device=device)
            preds = []

            for i in range(L_total):
                new_slot = torch.stack([
                    prices_batch[:, i],
                    torch.full((B,), 0.5, device=device),
                ], dim=-1).unsqueeze(1)                          # (B, 1, 2)
                input_seq = torch.cat([input_seq, new_slot], dim=1)

                p_i, logit_i, _ = model(input_seq)               # (B,), (B,), _
                preds.append((p_i, logit_i))

                
                if i < L_total - 1:
                    updated = input_seq.clone()
                    updated[:, i, 1] = p_i
                    input_seq = updated

            
            logits = torch.stack([lo for _, lo in preds], dim=1)         # (B, T+1)
            probs = torch.stack([p for p, _ in preds], dim=1)            # (B, T+1)
            loss = F.binary_cross_entropy_with_logits(logits, true_probs_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            with torch.no_grad():
                mae = (probs - true_probs_batch).abs().mean().item()

            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"epoch {epoch} step {global_step:4d} | "
                    f"bce {loss.item():.4f} | "
                    f"mae {mae:.4f} | "
                    f"lr {sched.get_last_lr()[0]:.2e}"
                )

    return model


if __name__ == "__main__":
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path.resolve()}")

    cfg = load_config(str(cfg_path))
    print(f"[arch3] loaded config from {cfg_path.resolve()}")

    model = train(cfg)

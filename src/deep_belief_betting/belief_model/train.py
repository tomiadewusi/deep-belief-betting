from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deep_belief_betting.belief_model.data import load_config, PriceDataset
from deep_belief_betting.belief_model.model import Architecture3
from deep_belief_betting.belief_model.device import resolve_device

def train(cfg: SimpleNamespace) -> Architecture3:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_set = PriceDataset(csv_path=cfg.data_path, T=cfg.T)
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

    enable_true_prob_head = cfg.enable_true_prob_head
    true_prob_loss_weight = cfg.true_prob_loss_weight

    for epoch in range(cfg.n_epochs):
        for features_batch, targets_batch, terminal_labels_batch in loader:
            features_batch = features_batch.to(device)       # (B, T+1, 2)
            targets_batch = targets_batch.to(device)         # (B, T+1)
            terminal_labels_batch = terminal_labels_batch.to(device)  # (B,)
            B = features_batch.shape[0]

            logits = []
            outcome_logits = []
            for i in range(L_total):
                x = features_batch[:, :i+1, :]               # (B, i+1, 2)
                _, logit_i, _, out_i = model(x)              # (B,), optional (B,)
                logits.append(logit_i)
                if enable_true_prob_head:
                    outcome_logits.append(out_i)

            logits = torch.stack(logits, dim=1)               # (B, T+1)
            if enable_true_prob_head:
                outcome_logits = torch.stack(outcome_logits, dim=1)   # (B, T+1)
                terminal_targets = terminal_labels_batch.unsqueeze(1).expand(-1, L_total)  # (B, T+1)
                latent_loss = F.binary_cross_entropy_with_logits(logits, targets_batch)
                outcome_loss = F.binary_cross_entropy_with_logits(outcome_logits, terminal_targets)
                loss = (1 - true_prob_loss_weight) * latent_loss + true_prob_loss_weight * outcome_loss
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            with torch.no_grad():
                preds = torch.sigmoid(logits)
                mae = (preds - targets_batch).abs().mean().item()

            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"epoch {epoch} step {global_step:4d} | "
                    f"bce {loss.item():.4f} | "
                    f"mae {mae:.4f} | "
                    f"lr {sched.get_last_lr()[0]:.2e}"
                )

    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        project_root = Path(__file__).parent.parent.parent
        ck_path = project_root / cfg.checkpoint_path
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, ck_path)
        print(f"[arch3] saved checkpoint → {ck_path}")

    return model


if __name__ == "__main__":
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path.resolve()}")

    cfg = load_config(str(cfg_path))
    print(f"[arch3] loaded config from {cfg_path.resolve()}")

    model = train(cfg)

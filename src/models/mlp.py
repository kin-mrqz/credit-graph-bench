"""MLP baseline with early stopping and epoch checkpoints. Accepts soft targets."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..ckpt import TrainingCheckpoint
from ..eval.metrics import auc


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.body(x)
        return self.head(h).squeeze(-1), h


def _batches(n: int, bs: int, shuffle: bool, gen: torch.Generator | None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: dict,
            ckpt: TrainingCheckpoint, seed: int = 0, device=None, pos_weight: float | None = None,
            log=print) -> tuple[MLP, list[dict]]:
    """X_* must already be prepared (TabularPrep). y_tr may be soft in [0, 1]; y_val is 0/1."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP(X_tr.shape[1], cfg["hidden"], cfg.get("dropout", 0.2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y_tr, dtype=np.float32))
    Xv = torch.as_tensor(X_val, dtype=torch.float32).to(device)

    state = ckpt.load(model, opt, map_location=device)
    start, best, bad = 1, -np.inf, 0
    if state:
        start, best, bad = state["epoch"] + 1, state["best_metric"], state["extra"].get("bad_epochs", 0)
        log(f"[mlp] resumed at epoch {start} (best val auc {best:.4f})")
    history = ckpt.history()
    bs, epochs, patience = cfg.get("batch_size", 1024), cfg["epochs"], cfg.get("patience", 5)

    for epoch in range(start, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in _batches(len(Xt), bs, True, gen):
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            logit, _ = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        p_val = predict_mlp(model, Xv, device)
        val_auc = auc(y_val, p_val)
        improved = val_auc > best
        if improved:
            best, bad = val_auc, 0
            ckpt.save_best(model)
        else:
            bad += 1
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1), "val_auc": val_auc, "best": best}
        history.append(rec); ckpt.append_history(rec)
        ckpt.save(model, opt, epoch, best, extra={"bad_epochs": bad})
        log(f"[mlp] epoch {epoch:3d} loss {rec['train_loss']:.4f} val_auc {val_auc:.4f}{' *' if improved else ''}")
        if bad >= patience:
            log("[mlp] early stop")
            break
    ckpt.save(model, opt, epoch, best, extra={"bad_epochs": bad}, force=True)
    ckpt.load_best(model, map_location=device)
    return model, history


@torch.no_grad()
def predict_mlp(model: MLP, X, device=None, bs: int = 16384) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(np.asarray(X, dtype=np.float32)) if not torch.is_tensor(X) else X
    out = []
    for i in range(0, len(X), bs):
        logit, _ = model(X[i : i + bs].to(device))
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

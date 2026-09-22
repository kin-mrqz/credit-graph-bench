"""Mini-batch GNN training with neighbour sampling, early stopping, and epoch checkpoints.

Works for homogeneous graphs (``make_data``) and homogenised heterogeneous graphs (``make_hetero_data``).
Loss is computed on seed nodes only (``batch.batch_size`` first rows), so unlabelled nodes can sit in
the graph without a label.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader

from ..ckpt import TrainingCheckpoint
from ..eval.metrics import auc
from ..graphs.base import EdgeList
from ..graphs.relational import RELATIONS


# ----------------------------------------------------------------------------- data construction
def make_data(X: np.ndarray, y: np.ndarray, edges: EdgeList, symmetric: bool = True) -> Data:
    e = edges.symmetrized() if symmetric and not edges.meta.get("symmetrized") else edges
    data = Data(
        x=torch.as_tensor(np.asarray(X, dtype=np.float32)),
        y=torch.as_tensor(np.asarray(y, dtype=np.float32)),   # NaN for unlabelled
        edge_index=torch.as_tensor(e.edge_index()),
    )
    if e.edge_type is not None:
        data.edge_type = torch.as_tensor(e.edge_type.astype(np.int64))
    data.num_nodes = edges.n_nodes
    return data


def make_hetero_data(payload: dict, y_customer: np.ndarray) -> tuple[Data, list[int], int]:
    """Homogenise the Home Credit hetero payload. Customer nodes come first (ids 0..Nc-1).

    Returns (data, in_dims per node type, num_relations). Reverse relations are added.
    """
    hd = HeteroData()
    types = ["customer", "bureau", "prev_app"]
    in_dims = [int(payload[f"x_{t}"].shape[1]) for t in types]
    width = max(in_dims)
    for t in types:
        x = payload[f"x_{t}"]
        if x.shape[1] < width:
            x = np.concatenate([x, np.zeros((x.shape[0], width - x.shape[1]), np.float32)], axis=1)
        hd[t].x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        n = x.shape[0]
        hd[t].y = torch.as_tensor(np.asarray(y_customer, dtype=np.float32)) if t == "customer" else torch.full((n,), float("nan"))
    for src_t, rel, dst_t in RELATIONS:
        ei = torch.as_tensor(payload[f"ei_customer__{rel}"].astype(np.int64))
        hd[src_t, rel, dst_t].edge_index = ei
        hd[dst_t, f"rev_{rel}", src_t].edge_index = ei.flip(0)
    data = hd.to_homogeneous(node_attrs=["x", "y"], add_node_type=True, add_edge_type=True)
    num_relations = len(hd.edge_types)
    return data, in_dims, num_relations


# ----------------------------------------------------------------------------- training
def _loader(data: Data, idx: np.ndarray, cfg: dict, shuffle: bool, batch_size: int | None = None) -> NeighborLoader:
    return NeighborLoader(
        data,
        num_neighbors=list(cfg["fanout"]),
        input_nodes=torch.as_tensor(np.asarray(idx, dtype=np.int64)),
        batch_size=batch_size or cfg["batch_size"],
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 0)),
        persistent_workers=False,
    )


def train_gnn(model: nn.Module, data: Data, train_idx: np.ndarray, val_idx: np.ndarray, cfg: dict,
              ckpt: TrainingCheckpoint, seed: int = 0, device=None, pos_weight: float | None = None,
              log=print) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    state = ckpt.load(model, opt, map_location=device)
    start, best, bad = 1, -np.inf, 0
    if state:
        start, best, bad = state["epoch"] + 1, state["best_metric"], state["extra"].get("bad_epochs", 0)
        log(f"[gnn] resumed at epoch {start} (best val auc {best:.4f})")
    history = ckpt.history()
    epochs, patience = cfg["epochs"], cfg.get("patience", 5)
    y_val = data.y[torch.as_tensor(val_idx)].numpy()

    train_loader = _loader(data, train_idx, cfg, shuffle=True)
    epoch = start - 1
    for epoch in range(start, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logit, _ = model(batch)
            bs = batch.batch_size
            loss = loss_fn(logit[:bs], batch.y[:bs])
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        logits_val, _ = predict_gnn(model, data, val_idx, cfg, device, want_emb=False)
        val_auc = auc(y_val, 1 / (1 + np.exp(-logits_val)))
        improved = val_auc > best
        if improved:
            best, bad = val_auc, 0
            ckpt.save_best(model)
        else:
            bad += 1
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1), "val_auc": float(val_auc), "best": float(best)}
        history.append(rec); ckpt.append_history(rec)
        ckpt.save(model, opt, epoch, best, extra={"bad_epochs": bad})
        log(f"[gnn] epoch {epoch:3d} loss {rec['train_loss']:.4f} val_auc {val_auc:.4f}{' *' if improved else ''}")
        if bad >= patience:
            log("[gnn] early stop")
            break
    ckpt.save(model, opt, epoch, best, extra={"bad_epochs": bad}, force=True)
    ckpt.load_best(model, map_location=device)
    return history


@torch.no_grad()
def predict_gnn(model: nn.Module, data: Data, idx: np.ndarray, cfg: dict, device=None, want_emb: bool = True,
                batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Logits (and embeddings) for `idx`, in the order of `idx`. Uses the same fanout as training."""
    device = device or next(model.parameters()).device
    model.eval()
    loader = _loader(data, idx, cfg, shuffle=False, batch_size=batch_size or cfg["batch_size"] * 2)
    logits, embs = [], []
    for batch in loader:
        batch = batch.to(device)
        logit, emb = model(batch)
        bs = batch.batch_size
        logits.append(logit[:bs].float().cpu().numpy())
        if want_emb:
            embs.append(emb[:bs].half().cpu().numpy())
    return np.concatenate(logits), (np.concatenate(embs) if want_emb else None)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))).astype(np.float32)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

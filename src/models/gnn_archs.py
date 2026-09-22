"""GNN architectures. Every model takes a PyG (mini-)batch and returns (logits, embeddings).

    logits:     (N,)   default logit per node in the batch
    embeddings: (N, emb_dim) penultimate representation (used by the hybrid and as teacher features)
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, RGCNConv, SAGEConv


class _Homo(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int, layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden] * (layers - 1) + [emb_dim]
        for i in range(layers):
            self.convs.append(self.make_conv(dims[i], dims[i + 1], last=(i == layers - 1)))
            if i < layers - 1:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
        self.head = nn.Linear(emb_dim, 1)

    def make_conv(self, i, o, last):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, batch):
        x, ei = batch.x, batch.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, ei)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, self.training)
        emb = x
        return self.head(emb).squeeze(-1), emb


class SAGE(_Homo):
    def make_conv(self, i, o, last):
        return SAGEConv(i, o, aggr="mean")


class GAT(_Homo):
    def __init__(self, in_dim, hidden, emb_dim, layers=2, dropout=0.3, heads=4):
        self.heads = heads
        super().__init__(in_dim, hidden, emb_dim, layers, dropout)

    def make_conv(self, i, o, last):
        if last:
            return GATConv(i, o, heads=1, concat=False, dropout=self.dropout)
        assert o % self.heads == 0, "hidden must be divisible by heads"
        return GATConv(i, o // self.heads, heads=self.heads, concat=True, dropout=self.dropout)


class HeteroRGCN(nn.Module):
    """RGCN over a homogenised heterogeneous graph (PyG ``HeteroData.to_homogeneous()``).

    Node features of different types are zero-padded to a common width before homogenisation;
    a per-type linear layer projects them to ``hidden`` so the padding does not matter.
    """

    def __init__(self, in_dims: list[int], hidden: int, emb_dim: int, num_relations: int, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.in_dims = in_dims
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in in_dims])
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [hidden] * layers + [emb_dim]
        for i in range(layers):
            self.convs.append(RGCNConv(dims[i], dims[i + 1], num_relations=num_relations, num_bases=None))
            if i < layers - 1:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, batch):
        x_in, nt = batch.x, batch.node_type
        h = x_in.new_zeros(x_in.shape[0], self.proj[0].out_features)
        for t, lin in enumerate(self.proj):
            m = nt == t
            if m.any():
                h[m] = lin(x_in[m, : self.in_dims[t]])
        h = torch.relu(h)
        for i, conv in enumerate(self.convs):
            h = conv(h, batch.edge_index, batch.edge_type)
            if i < len(self.convs) - 1:
                h = self.norms[i](h)
                h = torch.relu(h)
                h = nn.functional.dropout(h, self.dropout, self.training)
        return self.head(h).squeeze(-1), h


def build_model(arch: str, in_dim: int, cfg: dict, **kw) -> nn.Module:
    hidden, emb, layers, dropout = cfg["hidden"], cfg.get("emb_dim", 64), cfg.get("layers", 2), cfg.get("dropout", 0.3)
    if arch == "sage":
        return SAGE(in_dim, hidden, emb, layers, dropout)
    if arch == "gat":
        return GAT(in_dim, hidden, emb, layers, dropout, heads=cfg.get("heads", 4))
    if arch == "rgcn":
        return HeteroRGCN(kw["in_dims"], hidden, emb, kw["num_relations"], layers, dropout)
    raise ValueError(f"unknown arch '{arch}'")

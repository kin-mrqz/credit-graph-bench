"""GAT wrapper. Architecture lives in gnn_archs so torch is imported in one place."""

from .gnn_archs import GAT

__all__ = ["GAT"]

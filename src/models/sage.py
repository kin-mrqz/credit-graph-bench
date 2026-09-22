"""GraphSAGE wrapper. Architecture lives in gnn_archs so torch is imported in one place."""

from .gnn_archs import SAGE

__all__ = ["SAGE"]

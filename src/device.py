"""Pick compute devices without forcing a torch import on machines that lack it."""

from __future__ import annotations

import os


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def torch_device():
    import torch

    forced = os.environ.get("TORCH_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xgb_device() -> str:
    """XGBoost >= 2.0 takes ``device='cuda'`` or ``'cpu'``."""
    forced = os.environ.get("XGB_DEVICE")
    if forced:
        return forced
    return "cuda" if cuda_available() else "cpu"


def summary() -> dict:
    info = {"torch": None, "cuda": False, "gpu": None, "pyg": None, "xgb_device": xgb_device()}
    if torch_available():
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
        try:
            import torch_geometric

            info["pyg"] = torch_geometric.__version__
        except ImportError:
            pass
    return info

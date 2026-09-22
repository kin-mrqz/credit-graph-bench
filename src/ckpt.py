"""Checkpoint and resume helpers.

Three levels, matching the plan:

1. Unit markers. A unit is one notebook x dataset x graph_builder x seed and owns one run_dir.
   ``mark_done`` writes DONE with a summary; scripts/run_nb.sh writes FAILED on a non-zero exit
   and skips units whose DONE exists.

2. Training checkpoints. ``TrainingCheckpoint`` saves model + optimiser + epoch + best metric every
   N epochs and keeps the best weights separately. On restart, ``load`` returns the saved state so
   the loop continues from ``epoch + 1``.

3. Grid checkpoints. ``GridCheckpoint`` records one JSON entry per finished grid point (a sweep over
   lambda, T, k, ...). ``pending`` yields only the unfinished points.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

DONE = "DONE"
FAILED = "FAILED"


# ----------------------------------------------------------------------------- unit markers
def is_done(run_dir: Path) -> bool:
    return (Path(run_dir) / DONE).exists()


def mark_done(run_dir: Path, summary: dict[str, Any] | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / FAILED).unlink(missing_ok=True)
    payload = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(summary or {})}
    path = run_dir / DONE
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def mark_failed(run_dir: Path, exc: BaseException | str | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    text = exc if isinstance(exc, str) else "".join(traceback.format_exception(exc)) if exc else "unknown error"
    path = run_dir / FAILED
    path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}", encoding="utf-8")
    return path


def clear_markers(run_dir: Path) -> None:
    for name in (DONE, FAILED):
        (Path(run_dir) / name).unlink(missing_ok=True)


def read_done(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / DONE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ----------------------------------------------------------------------------- training checkpoints
class TrainingCheckpoint:
    """Epoch-level checkpoint for torch training loops.

    Files under ``run_dir/ckpt/<tag>/``:
        last.pt   model + optimiser + epoch + best metric + extra   (overwritten every `every` epochs)
        best.pt   model weights at the best validation metric
        history.json  list of per-epoch dicts
    """

    def __init__(self, run_dir: Path, tag: str = "model", every: int = 1):
        self.dir = Path(run_dir) / "ckpt" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every))
        self.last_path = self.dir / "last.pt"
        self.best_path = self.dir / "best.pt"
        self.history_path = self.dir / "history.json"

    # -- state
    def load(self, model, optimizer=None, map_location=None) -> dict[str, Any] | None:
        """Restore model (and optimiser) from last.pt. Returns the saved state dict or None."""
        if not self.last_path.exists():
            return None
        import torch

        state = torch.load(self.last_path, map_location=map_location or "cpu")
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        return state

    def save(self, model, optimizer, epoch: int, best_metric: float, extra: dict | None = None, force: bool = False) -> None:
        if not force and (epoch % self.every) != 0:
            return
        import torch

        tmp = self.last_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "epoch": int(epoch),
                "best_metric": float(best_metric),
                "extra": extra or {},
                "saved_at": time.time(),
            },
            tmp,
        )
        tmp.replace(self.last_path)  # atomic on POSIX; avoids a half-written checkpoint

    def save_best(self, model) -> None:
        import torch

        tmp = self.best_path.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(self.best_path)

    def load_best(self, model, map_location=None) -> bool:
        if not self.best_path.exists():
            return False
        import torch

        model.load_state_dict(torch.load(self.best_path, map_location=map_location or "cpu"))
        return True

    # -- history
    def history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def append_history(self, record: dict[str, Any]) -> None:
        hist = self.history()
        hist.append(record)
        self.history_path.write_text(json.dumps(hist, indent=1, default=float), encoding="utf-8")


# ----------------------------------------------------------------------------- grid checkpoints
class GridCheckpoint:
    """Per-grid-point results in one JSON file. Keys are stable strings, e.g. 'knn|k=10|lam=0.5|T=2'."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def key(**kwargs: Any) -> str:
        return "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))

    def done(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def record(self, key: str, result: dict[str, Any]) -> None:
        self._data[key] = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), **result}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, default=float), encoding="utf-8")
        tmp.replace(self.path)

    def pending(self, keys: Iterable[str]) -> Iterator[str]:
        for k in keys:
            if not self.done(k):
                yield k

    def results(self) -> dict[str, Any]:
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

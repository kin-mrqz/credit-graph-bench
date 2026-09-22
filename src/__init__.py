"""credit-graph-bench: benchmark of tabular, graph, and hybrid credit-default models.

The package is intentionally light to import. Torch-dependent modules live under
``src.models`` and are imported only by the notebooks that need them, so
``python -c "import src"`` works on a laptop without CUDA or PyTorch.
"""

__version__ = "0.1.0"

from . import config, paths  # noqa: F401  (light, no heavy deps)

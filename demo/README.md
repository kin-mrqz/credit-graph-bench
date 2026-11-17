# demo

A standalone notebook that runs the graph stages of the pipeline (notebooks 03-06) on the Taiwan
dataset on CPU, with interactive plots. It is separate from the pipeline in `notebooks/` and does
not write anything to `runs/`.

- `graph_demo.ipynb`: the notebook
- `demo_kit.py`: plotting and widget helpers; all modelling code comes from `src/`

## Running

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r demo/requirements.txt
jupyter lab demo/graph_demo.ipynb
```

Then run all cells. The first run downloads the Taiwan data (UCI, via `ucimlrepo`) into `data_raw/`
and computes a t-SNE map for the plots (cached in `demo/.cache/`), which takes a few minutes.
After that a full run takes about 90 seconds.

DGraph-Fin, Home Credit and Lending Club are too large to run here; the notebook reads their
results from `report/`.

## Notes

- The GraphSAGE model in section 05 is a plain-torch version of PyG's `SAGEConv(aggr="mean")`,
  trained full-batch, so the demo doesn't need PyG or a GPU.
- `demo_kit` blocks the `optree` import before torch loads. On my Windows setup, importing it
  inside a Jupyter kernel crashes the kernel. Import `demo_kit` before `torch`.
- Widget outputs only work with a live kernel. If a cell shows "Error displaying widget",
  restart the kernel and run all cells.

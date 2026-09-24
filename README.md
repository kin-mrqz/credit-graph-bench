# credit-graph-bench

Gate A benchmark: do graph models improve credit-default ranking enough, under a leakage-safe protocol, to justify a bank pilot?

This repository is the code and documentation. Notebooks are written locally and **not executed** on a laptop. They are executed on a GPU node by scripts, detached from Cursor, with checkpoints so a failure costs one unit of work.

## What this is

A credit risk engine scores one loan application and returns a default probability, a score, and reasons. Production engines are judged by time-split ranking quality, calibration, stability, and business translation (approval at a fixed bad rate), not by a single AUC on a random split.

This project compares four model families on four public datasets:

| Tier | Family | Inference needs a graph? |
|---|---|---|
| 0 | Logistic regression, XGBoost, MLP (tabular) | no |
| 1 | Hand-built graph features into XGBoost | no (features precomputed) |
| 2 | GraphSAGE, GAT, RGCN | yes |
| 3 | GNN embedding concatenated into XGBoost | no at scoring if embeddings are cached; yes to produce them |
| 4 | LUPI student: XGBoost trained on teacher soft labels, tabular only at test | no |

Datasets: DGraph-Fin (real emergency-contact graph, repayment proxy), Home Credit (relational history graph), Lending Club (out-of-time split; shared-attribute / kNN graphs), Taiwan UCI (synthetic kNN only).

Pre-registered success: a family is called "lift" only if AUC **and** PR-AUC beat XGBoost with a 95% CI excluding zero on at least two datasets, without losing calibration or stability. The hybrid (Tier 3) is the commercial candidate. LUPI transfers graph structure only if the student also beats the random-graph and self-distillation controls.

This does **not** produce a calibrated PD for a real bank portfolio.

## Layout

```
notebooks/   numbered stages (the deliverable)
src/         shared loaders, graphs, models, metrics, checkpoints
scripts/     GPU setup, fetch, papermill runner, SLURM / tmux detach
configs/     one YAML per dataset plus stage unit list
splits/      frozen train/val/test indices (written by notebook 01, then committed)
runs/        gitignored outputs: executed.ipynb, DONE/FAILED, predictions, ckpts
report/      HTML/PDF of 09_report.ipynb and commercial_summary.md
```

## Local build (laptop, no GPU, no notebook execution)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/lint.py          # or: make lint
```

`lint` runs ruff, import-checks the torch-free parts of `src`, byte-compiles the rest, and checks that every notebook has a papermill `parameters` cell and a `ckpt.mark_done` call. It does not execute cells.

Install the git filter so committed notebooks stay output-free:

```bash
nbstripout --install --attributes .gitattributes
```

Push to GitHub, then pull on the GPU node.

## GPU node (HKU GPU Farm or any CUDA box)

```bash
git clone <this-repo> && cd credit-graph-bench
bash scripts/setup_env.sh          # CUDA PyTorch + PyG + requirements.txt
source scripts/env.sh
bash scripts/fetch_data.sh         # Taiwan automatic; Home Credit / Lending Club need Kaggle creds
# Place DGraphFin.zip under $DATA_ROOT/dgraphfin/raw/ (licence click-through)
```

Detached execution (survives SSH drop):

```bash
bash scripts/run_detached.sh                 # tmux or nohup around run_all.sh
# or SLURM:
bash scripts/sbatch/submit_chain.sh
```

One unit:

```bash
bash scripts/run_nb.sh 02_baselines_tabular dataset=taiwan seed=0
```

Already-DONE units are skipped. Failed units leave a `FAILED` file; rerun the same command. GNN training resumes from `runs/.../ckpt/<tag>/last.pt`. Sweeps resume from `grid.json`.

Smoke test on subsampled data:

```bash
QUICK=1 bash scripts/run_all.sh
```

## Environment variables

| Variable | Meaning |
|---|---|
| `DATA_ROOT` | Raw and converted data (default `./data_raw`) |
| `RUNS_ROOT` | Outputs (default `./runs`) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API |
| `QUICK=1` | Subsample + few epochs |
| `CUDA` | Wheel tag for `setup_env.sh` (`cu121`, `cu124`, `cpu`) |

## Storage

Plan for ~30–45 GB on the node (CUDA env + converted data + graphs + 5-seed artefacts). Keep raw zips deleted after conversion (`KEEP_ARCHIVES=0`). Laptop 35 GB is enough for lint and editing, not for DGraph-Fin training.

## Licence and data

Code in this repo is for the Gate A study. The four datasets have their own licences; you must accept them at the source (UCI, Kaggle, DGraph).

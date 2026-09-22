PYTHON ?= python
STAGES := 00_env_check 01_load_and_split 02_baselines_tabular 03_build_graphs 04_graph_features_xgb 05_gnn 06_hybrid 07_lupi_distill 08_calibration_stability_business 90_report

.PHONY: lint strip test env data run-all report clean-markers

# Local: syntax, imports, notebook structure. Executes no notebook cell.
lint:
	$(PYTHON) scripts/lint.py

# Local: clear notebook outputs in place (the git filter does this on commit as well).
strip:
	nbstripout notebooks/*.ipynb

# Local: leakage and graph-builder unit tests (no data download)
test:
	$(PYTHON) -m pytest tests -q

# Node: environment and data
env:
	bash scripts/setup_env.sh

data:
	bash scripts/fetch_data.sh

# Node: run every stage in order, stop at first stage with a FAILED unit
run-all:
	bash scripts/run_all.sh

# Node: export the report notebook to HTML
report:
	bash scripts/run_nb.sh 90_report dataset=all graph_builder=all seed=0 force=1
	jupyter nbconvert --to html "$${RUNS_ROOT:-runs}/90_report/all/all/0/executed.ipynb" --output-dir report --output gate_a_report

# Node: remove FAILED markers so run_all reruns only those units
clean-markers:
	find "$${RUNS_ROOT:-runs}" -name FAILED -delete

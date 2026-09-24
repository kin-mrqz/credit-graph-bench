# SLURM job files (HKU CS GPU Farm)

One file per stage. Each requests one RTX 4080 on the `batch` partition (`--gres=gpu:rtx_4080:1`), 4 CPUs (`RestrictedCoresPerGPU`), and at most 40G memory, which is the largest request this account can submit. It activates the environment via `scripts/env.sh`,
and calls `scripts/run_stage.sh <stage>`. Logs go to `$RUNS_ROOT/slurm/`.

Submit one stage:

    sbatch scripts/sbatch/05_gnn.sbatch

Submit everything as a dependency chain (each stage starts after the previous one succeeds):

    bash scripts/sbatch/submit_chain.sh

Rerun only failed units of a stage (DONE units are skipped automatically):

    sbatch scripts/sbatch/05_gnn.sbatch

Resource sizing follows the storage and runtime estimates in the plan. Adjust `--time`,
`--mem`, and `--partition` to the farm's current limits (`sinfo`, `scontrol show partition`).
If the farm caps you at 2 GPUs concurrently, do not submit more than two heavy stages at once.

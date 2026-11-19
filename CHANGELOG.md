# Changelog

## 0.1.0 (2026-11-19)

First release.

- Shows, for every configured cluster, the compute budgets (remaining until the end of the project
  and, where budgets are split per month, until the end of the month), the storage quotas and the
  user's running and pending jobs.
- Supports LUMI, JUPITER, Leonardo and MeluXina, plus the jobs of any other Slurm cluster.
- Shows the GPUs of the nodes of every cluster and estimates every budget that can be spent on
  them in GPU-hours, with a summary of the GPU-hours left on all clusters.
- One SSH session per cluster, all clusters queried in parallel, with a time limit per cluster.
- TOML configuration (`~/.config/hpcmon/config.toml`) and JSON output (`--json`).

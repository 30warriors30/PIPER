# Experiment pipeline

The OPT-1.3B/C4 command, input preparation, and output fields are documented in [PAPER_EXPERIMENTS.md](../PAPER_EXPERIMENTS.md). Start there for the `PIPER_selfhash` headline run.

`run_pareto_sweep.py` builds a fixed manifest, generates shared clean and natural continuations, generates watermarked continuations for the selected operating point, detects all text classes, scores continuation quality, and aggregates the results. `--preset paper` selects `soft_p0_m2`. The `smoke` and `full` presets select diagnostic and bias-sweep points.

Stages can be run separately with `--stage manifest`, `baseline`, `sweep`, and `aggregate`. Keep all experiment arguments and the experiment ID the same across stages. Use `--resume` after interruption. The initial `--stage all` command in [PAPER_EXPERIMENTS.md](../PAPER_EXPERIMENTS.md) is simpler for a first run.

The run directory contains `experiment.json`, `manifest.jsonl`, `shared/` null outputs, `runs/<point_id>/` watermarked outputs, and aggregate CSV/JSON files. The exact-binomial PIPER path does not estimate a Z-score threshold. Calibration files are used only when `--presence-test z_score` is explicitly selected.

Other modules in this directory support the additional evaluations developed with this source tree. External comparison repositories and their output files are not bundled here.

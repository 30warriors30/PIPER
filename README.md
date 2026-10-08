# PIPER

This is the PIPER selfhash implementation used for the reported OPT-1.3B/C4 experiment. Model weights, datasets, generated samples, and the separate baseline implementations are not included.

PIPER biases a selected child of a keyed two-level token partition. Detection first tests parent membership with an exact binomial tail; payload voting and BCH decoding run only after that presence gate accepts. The released selfhash path conditions a candidate's partition on the candidate token and recent history. The `history` path remains in the code for older runs.

## Headline configuration

| Setting | Value |
| --- | --- |
| Generator and prompts | OPT-1.3B, C4 RealNewsLike validation |
| Continuation | 300 tokens; temperature 1.0; top-k 50; top-p 0.95 |
| Watermark | `soft_p0_m2`: parent bias 0, payload-child bias 2 |
| Partition | `selfhash`, `v2`, raw-logit candidate top-k 50 |
| Payload | 8 bits, BCH(23, 8, 3) |
| Presence test | Exact binomial, alpha 0.01, unique-context counting |
| Reporting | Blind-text detection; tie-to-zero payload decoding |
| Samples | 1,000 watermarked tests; 5,000 model and natural null tests |

The experiment uses a separate 1,000-record calibration split for methods that need calibration. The PIPER exact-binomial decision does not fit or read a threshold from that split. Other operating points and comparison methods are outside this headline run.

## Install

Use Python 3.10 and a CUDA-capable GPU. The supplied Conda environment pins the research dependencies, including PyTorch 2.8.0 with CUDA 12.8 wheels:

```bash
conda env create -f environment.yml
conda activate dual-watermark-blackwell
python -m pip check
pytest -q
```

The code is not limited to Blackwell GPUs. Check that the PyTorch build supports your GPU before running generation. A CPU-only dry run is available below.

## Prepare inputs

Download an OPT-1.3B checkpoint and set `MODEL_PATH` to its local directory. The preparation command reads C4 RealNewsLike validation through the Hugging Face Dataset Viewer; pass `--source-parquet` to use a local Parquet file instead. It writes processed JSONL and metadata, not model weights.

```bash
export MODEL_PATH=/path/to/opt-1.3b
mkdir -p data
python -m experiments.prepare_c4_realnewslike \
  --model-path "$MODEL_PATH" \
  --output-path data/c4_realnewslike_validation_t300_6000.jsonl \
  --prompt-tokens 30 --continuation-tokens 300 --records 6000
export C4_PATH="$PWD/data/c4_realnewslike_validation_t300_6000.jsonl"
```

The preprocessing result depends on the dataset snapshot and model tokenizer. Record both revisions if you need exact numerical replication. The original data manifest and model checkpoint are not part of this source release.

For a three-sample pipeline check, run `bash scripts/smoke_test.sh` after setting `MODEL_PATH` and `C4_PATH`. Its 64-token outputs are for checking the installation, not for comparing with the paper.

## Run the headline experiment

The wrapper sets every headline parameter explicitly. `DRY_RUN=1` prints the configuration without loading a model or writing results.

```bash
export SECRET_KEY=dual-layer-key-2026
DRY_RUN=1 bash scripts/run_pareto_sweep.sh
bash scripts/run_pareto_sweep.sh
```

To resume an interrupted run, execute the same command again. Set `OVERWRITE=1` only when intentionally replacing its existing output. Set `BATCH_SIZE` to fit available GPU memory. The run directory is `outputs/experiments/piper_main_t300_b8_selfhash_top50/`.

Read `runs/soft_p0_m2/metrics.json` for the watermarked TPR and correct attribution rate. `shared/negative_detections.jsonl` contains the null detections; `aggregate_summary.json` and `sweep_results.csv` summarize the run. The headline input mode is `blind_text`. The external OPT-6.7B perplexity calculation is described in [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md).

## Repository map

- `watermark/`: keyed partition, logits processor, presence detector, and BCH codec.
- `utils/`: model, dataset, generation, and file handling.
- `experiments/`: manifest, generation, detection, quality, and aggregation pipeline.
- `evaluation/`: metrics and reports for the direct Python entry points.
- `tests/`: unit and small integration tests that do not require a model download.
- `run_generation.py`, `run_detection.py`, `evaluate_results.py`: direct entry points for smaller runs.

Generated outputs are not included. This source release does not contain results for the paper's baseline comparisons or every appendix experiment. See [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the exact command and output interpretation for the OPT-1.3B/C4 setting.

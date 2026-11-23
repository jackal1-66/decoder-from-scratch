# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-23

### Added

- `decoder_mlx`: an MLX implementation of the decoder, installed with the `mlx` extra
  (`pip install "decoder-from-scratch[mlx]"`). Same `Config` and parameter names as the
  PyTorch model; RoPE and attention written out explicitly and through `mx.fast` kernels
  (the default), KV cache, sampling with explicit PRNG keys, LoRA with merge and unmerge.
- MLX checkpoints: safetensors with the config and a format version in the metadata.
- `decoder-convert`: converts checkpoints between PyTorch (`.pt`) and MLX (`.safetensors`)
  in both directions, bit for bit, after validating config, names, shapes and dtypes;
  never leaves partial output.
- `python -m decoder_mlx.train`: pretraining and fine-tuning (full or LoRA) in MLX, with
  checkpointing and exact resume (`--resume`).
- `decoder-benchmark`: times PyTorch and MLX (forward pass, `mx.compile`, generation with
  and without the cache) and writes JSON and a Markdown table.
- `scripts/mlx_equivalence.py` and `scripts/compare_training.py`: the measurements behind
  the equivalence results in the README.
- `decoder.config` (framework-free `Config`, re-exported by `decoder.model`),
  `decoder.train.eval_batches`, and `TrainConfig.log_every`, which logs the training loss.
- CI jobs for the MLX tests, for the PyTorch package without MLX, and for the minimum
  supported versions (torch 2.4.1, numpy 2.0.2, mlx 0.32.4).

### Changed

- `Config` rejects non-positive sizes, `dropout` outside [0, 1) and non-positive
  `rope_base` or `norm_eps`, instead of failing later or not at all.
- `decoder.train` reports progress through `logging` instead of `print`, and `log.jsonl`
  also contains `train_bpb` records every `log_every` steps (default 50).
- CI installs torch from the CPU index only, so pip cannot pick a CUDA build from PyPI.
- Tests treat warnings as errors.

### Fixed

- `decoder.train` closes `log.jsonl` when training raises or is interrupted.

## [0.1.0]

### Added

- Decoder-only transformer in PyTorch (RMSNorm, RoPE, grouped-query attention, SwiGLU,
  tied head, KV cache), sampling, LoRA, byte-level training, property tests, and the
  generation and fine-tuning experiments.

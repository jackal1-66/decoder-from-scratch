"""Numerical agreement of the PyTorch and MLX models in float32, on CPU.

    python scripts/mlx_equivalence.py                       # random weights, default Config
    python scripts/mlx_equivalence.py --checkpoint runs/pretrain/model.pt --text shakespeare.txt

The PyTorch weights are converted to MLX and both models score the same token windows
(from the end of --text, else random tokens). Logit differences are reported against a
float64 PyTorch reference, which separates the two questions:

* torch_fp32 and mlx_fp32 vs fp64: the rounding error float32 itself causes in each
  implementation;
* torch_fp32 vs mlx_fp32: the difference between the implementations.

If the last is of the order of the first two, the implementations agree as closely as
float32 arithmetic allows. Greedy generation is also run in both; the report gives the
first position where the tokens differ (null if none). Prints JSON.
"""

import argparse
import dataclasses
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from decoder.config import Config
from decoder.model import Decoder
from decoder.sampling import generate as torch_generate
from decoder_mlx.convert import read_torch_checkpoint, torch_to_mlx
from decoder_mlx.model import Decoder as MLXDecoder
from decoder_mlx.sampling import generate as mlx_generate


def differences(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float64) - reference)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_abs_over_max_logit": float(diff.max() / np.abs(reference).max()),
    }


def load_models(checkpoint: Path | None) -> tuple[Decoder, Config]:
    if checkpoint is None:
        torch.manual_seed(0)
        config = Config()
        return Decoder(config).eval(), config
    config, state = read_torch_checkpoint(checkpoint)
    model = Decoder(config)
    model.load_state_dict(state)
    return model.eval(), config


def token_windows(config: Config, text: Path | None, batch: int) -> np.ndarray:
    T = config.max_seq_len
    if text is None:
        return np.random.default_rng(0).integers(0, config.vocab_size, (batch, T))
    data = np.frombuffer(text.read_bytes(), dtype=np.uint8)[-batch * T :]  # held-out tail
    return data.reshape(batch, T).astype(np.int64)


def first_difference(a: np.ndarray, b: np.ndarray) -> int | None:
    mismatch = np.nonzero((a != b).any(axis=0))[0]
    return int(mismatch[0]) if mismatch.size else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, help="decoder.train .pt file (default: random)")
    p.add_argument("--text", type=Path, help="score windows from the end of this text")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--new-tokens", type=int, default=200)
    args = p.parse_args()

    model, config = load_models(args.checkpoint)
    tokens = token_windows(config, args.text, args.batch)
    weights = torch_to_mlx(model.state_dict(), config)
    with torch.no_grad():
        torch32 = model(torch.from_numpy(tokens)).numpy()
        reference = model.double()(torch.from_numpy(tokens)).numpy()
        model.float()

    report: dict[str, Any] = {
        "config": dataclasses.asdict(config),
        "weights": str(args.checkpoint) if args.checkpoint else "random (torch seed 0)",
        "tokens": f"{args.batch} x {config.max_seq_len} "
        + ("from text" if args.text else "random"),
        "versions": {"torch": torch.__version__, "mlx": version("mlx"), "numpy": np.__version__},
        "machine": platform.machine(),
        "mlx_device": str(mx.default_device()),
        "max_abs_logit": float(np.abs(reference).max()),
        "torch_fp32_vs_fp64": differences(torch32, reference),
    }
    prompt = tokens[:, : config.max_seq_len - args.new_tokens]
    with torch.no_grad():
        torch_tokens = torch_generate(
            model, torch.from_numpy(prompt), args.new_tokens, temperature=0.0
        ).numpy()
    for fast in (True, False):
        mlx_model = MLXDecoder(config, use_fast_kernels=fast)
        mlx_model.load_weights(list(weights.items()))
        mlx_model.eval()
        mlx32 = np.array(mlx_model(mx.array(tokens.astype(np.int32))))
        mlx_tokens = np.array(
            mlx_generate(mlx_model, mx.array(prompt.astype(np.int32)), args.new_tokens, 0.0)
        )
        name = "mlx_fast" if fast else "mlx_explicit"
        report[f"{name}_fp32_vs_fp64"] = differences(mlx32, reference)
        report[f"torch_fp32_vs_{name}_fp32"] = differences(mlx32, torch32.astype(np.float64))
        report[f"greedy_{name}_first_difference"] = first_difference(
            mlx_tokens[:, prompt.shape[1] :], torch_tokens[:, prompt.shape[1] :]
        )
    report["greedy_tokens_compared"] = f"{args.batch} x {args.new_tokens}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

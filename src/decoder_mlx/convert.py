"""Convert checkpoints between the PyTorch and the MLX model.

    decoder-convert torch-to-mlx runs/pretrain/model.pt model.safetensors
    decoder-convert mlx-to-torch model.safetensors model.pt [--overwrite]

The PyTorch side is the ``.pt`` file written by ``decoder.train`` (a dict with ``config``
and ``model``, the ``state_dict``); the MLX side is the safetensors format of
``decoder_mlx.checkpoint``. Values and dtypes (float32, float16, bfloat16) are copied
exactly, so a round trip reproduces every tensor bit for bit.

Layouts: both frameworks store linear weights as (out, in) and embeddings as (V, C), so
nothing is transposed. The PyTorch model's ``head.weight`` is the same tensor as
``embed.weight``; it must equal it when converting to MLX, where the head is tied by
construction, and it is restored as a shared tensor when converting back.
"""

import argparse
import dataclasses
import logging
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from torch import Tensor

from decoder.config import Config
from decoder_mlx.checkpoint import (
    CheckpointError,
    check_tensors,
    config_from_dict,
    expected_shapes,
    load_weights,
    mlx_dtype_name,
    save_weights,
    write_atomically,
)

logger = logging.getLogger(__name__)

TIED_HEAD = "head.weight"


def _to_mlx(t: Tensor) -> mx.array:
    if t.dtype == torch.bfloat16:  # numpy has no bfloat16; bf16 -> fp32 -> bf16 is exact
        return mx.array(t.detach().float().numpy()).astype(mx.bfloat16)
    return mx.array(t.detach().numpy())


def _to_torch(a: mx.array) -> Tensor:
    if a.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def torch_to_mlx(
    state: Mapping[str, Tensor], config: Config, source: str = "PyTorch state_dict"
) -> dict[str, mx.array]:
    """MLX weights from a PyTorch ``Decoder.state_dict()``, validated against ``config``."""
    found = {n: (tuple(t.shape), str(t.dtype).removeprefix("torch.")) for n, t in state.items()}
    expected = expected_shapes(config) | {TIED_HEAD: (config.vocab_size, config.d_model)}
    check_tensors(found, expected, source)
    if not torch.equal(state[TIED_HEAD], state["embed.weight"]):
        raise CheckpointError(
            f"{source}: {TIED_HEAD} differs from embed.weight; the MLX model ties them and cannot "
            "represent an untied head"
        )
    return {name: _to_mlx(t.cpu()) for name, t in state.items() if name != TIED_HEAD}


def mlx_to_torch(weights: Mapping[str, mx.array], config: Config) -> dict[str, Tensor]:
    """A PyTorch ``state_dict`` from MLX weights; ``head.weight`` shares ``embed.weight``."""
    found = {n: (tuple(a.shape), mlx_dtype_name(a.dtype)) for n, a in weights.items()}
    check_tensors(found, expected_shapes(config), "MLX weights")
    state = {name: _to_torch(a) for name, a in weights.items()}
    state[TIED_HEAD] = state["embed.weight"]
    return state


def read_torch_checkpoint(path: Path) -> tuple[Config, dict[str, Tensor]]:
    """Config and ``state_dict`` of a ``.pt`` file written by ``decoder.train``."""
    try:
        # weights_only: refuse to unpickle anything but tensors and plain containers.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as err:
        raise CheckpointError(f"cannot read {path} as a PyTorch checkpoint: {err}") from err
    if not isinstance(checkpoint, dict) or not {"config", "model"} <= checkpoint.keys():
        raise CheckpointError(f"{path} is not a decoder.train checkpoint ('config', 'model')")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not all(isinstance(t, Tensor) for t in state.values()):
        raise CheckpointError(f"{path}: 'model' must map names to tensors")
    return config_from_dict(checkpoint["config"]), state


def write_torch_checkpoint(
    path: Path, config: Config, state: Mapping[str, Tensor], overwrite: bool = False
) -> None:
    checkpoint = {"config": dataclasses.asdict(config), "model": dict(state)}
    write_atomically(path, lambda tmp: torch.save(checkpoint, tmp), overwrite)


def convert_torch_to_mlx(src: Path, dst: Path, overwrite: bool = False) -> None:
    config, state = read_torch_checkpoint(src)
    save_weights(dst, config, torch_to_mlx(state, config, str(src)), overwrite)
    logger.info("wrote %s (%d tensors)", dst, len(state) - 1)


def convert_mlx_to_torch(src: Path, dst: Path, overwrite: bool = False) -> None:
    config, weights = load_weights(src)
    write_torch_checkpoint(dst, config, mlx_to_torch(weights, config), overwrite)
    logger.info("wrote %s (%d tensors + tied head)", dst, len(weights))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decoder-convert",
        description="Convert checkpoints between the PyTorch and MLX decoder models.",
    )
    directions = parser.add_subparsers(dest="direction", required=True)
    for name, help_text in (
        ("torch-to-mlx", "PyTorch .pt (from decoder.train) to MLX .safetensors"),
        ("mlx-to-torch", "MLX .safetensors to PyTorch .pt"),
    ):
        sub = directions.add_parser(name, help=help_text)
        sub.add_argument("src", type=Path)
        sub.add_argument("dst", type=Path)
        sub.add_argument("--overwrite", action="store_true", help="replace dst if it exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    convert = convert_torch_to_mlx if args.direction == "torch-to-mlx" else convert_mlx_to_torch
    try:
        convert(args.src, args.dst, overwrite=args.overwrite)
    except (CheckpointError, OSError) as err:
        parser.exit(1, f"decoder-convert: error: {err}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

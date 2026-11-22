"""MLX checkpoints: model weights and config in one safetensors file.

Format (``FORMAT_VERSION`` 1): one tensor per parameter, named as in the PyTorch
``state_dict`` (no ``head.weight``: the head is tied to ``embed.weight``), and string
metadata ``format`` = ``FORMAT_NAME``, ``format_version`` and ``config`` (JSON of
``decoder.config.Config``). Readers reject other formats and newer versions.

Every check runs before anything is written, and files are written to a temporary file
in the target directory and then renamed, so a failure never leaves partial output.
"""

import dataclasses
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from decoder.config import Config
from decoder_mlx.model import Decoder, flat_parameters

FORMAT_NAME = "decoder-from-scratch-mlx"
FORMAT_VERSION = 1
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


class CheckpointError(ValueError):
    """A checkpoint that cannot be read, converted or written as requested."""


def config_from_dict(raw: object) -> Config:
    """A ``Config`` from a checkpoint's config mapping; every field must be present."""
    if not isinstance(raw, Mapping):
        raise CheckpointError(f"config must be a mapping, got {type(raw).__name__}")
    fields = {f.name: type(f.default) for f in dataclasses.fields(Config)}
    missing, unknown = sorted(fields.keys() - raw.keys()), sorted(raw.keys() - fields.keys())
    if missing or unknown:
        raise CheckpointError(f"config fields do not match: missing {missing}, unknown {unknown}")
    for name, kind in fields.items():
        value = raw[name]
        # bool is an int subclass; ints are valid floats (JSON writes 10000.0 as 10000.0,
        # but hand-written configs may not).
        valid = (int, float) if kind is float else (int,)
        if isinstance(value, bool) or not isinstance(value, valid):
            raise CheckpointError(f"config field {name} must be {kind.__name__}, got {value!r}")
    try:
        return Config(**raw)
    except ValueError as err:
        raise CheckpointError(f"invalid config: {err}") from err


def expected_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    """Name and shape of every parameter of ``decoder_mlx.model.Decoder(config)``."""
    C, D, F = config.d_model, config.head_dim, config.d_ff
    shapes: dict[str, tuple[int, ...]] = {
        "embed.weight": (config.vocab_size, C),
        "norm.weight": (C,),
    }
    for i in range(config.n_layers):
        p = f"blocks.{i}."
        shapes |= {
            p + "attn_norm.weight": (C,),
            p + "attn.wq.weight": (config.n_heads * D, C),
            p + "attn.wk.weight": (config.n_kv_heads * D, C),
            p + "attn.wv.weight": (config.n_kv_heads * D, C),
            p + "attn.wo.weight": (C, config.n_heads * D),
            p + "ff_norm.weight": (C,),
            p + "ff.w1.weight": (F, C),
            p + "ff.w3.weight": (F, C),
            p + "ff.w2.weight": (C, F),
        }
    return shapes


def check_tensors(
    found: Mapping[str, tuple[tuple[int, ...], str]],
    expected: Mapping[str, tuple[int, ...]],
    source: str,
) -> None:
    """Raise one ``CheckpointError`` listing every name, shape and dtype problem.

    ``found`` maps each tensor name to its (shape, dtype name), e.g. ("float32").
    """
    problems = []
    missing = sorted(expected.keys() - found.keys())
    unexpected = sorted(found.keys() - expected.keys())
    if missing:
        problems.append(f"missing tensors: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected tensors: {', '.join(unexpected)}")
        if any(".lora_" in name for name in unexpected):
            problems.append("LoRA adapters must be merged into the base weights first")
    for name in sorted(expected.keys() & found.keys()):
        shape, dtype = found[name]
        if shape != expected[name]:
            problems.append(f"{name}: shape {shape}, the config implies {expected[name]}")
        if dtype not in SUPPORTED_DTYPES:
            problems.append(f"{name}: dtype {dtype} is not one of {', '.join(SUPPORTED_DTYPES)}")
    if problems:
        raise CheckpointError(f"{source} does not match its config:\n  " + "\n  ".join(problems))


def mlx_dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def write_atomically(dst: Path, write: Callable[[Path], None], overwrite: bool) -> None:
    """Call ``write(tmp)`` on a temporary file next to ``dst``, then rename it to ``dst``."""
    if dst.exists() and not overwrite:
        raise CheckpointError(f"{dst} already exists; pass overwrite=True (--overwrite)")
    if not dst.parent.is_dir():
        raise CheckpointError(f"output directory {dst.parent} does not exist")
    # Same directory, so the final rename is atomic; the suffix keeps dst's extension
    # because mx.save_safetensors appends ".safetensors" to paths without it.
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=dst.suffix)
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_weights(
    path: Path, config: Config, weights: Mapping[str, mx.array], overwrite: bool = False
) -> None:
    """Validate ``weights`` against ``config`` and write them as an MLX checkpoint."""
    if path.suffix != ".safetensors":
        raise CheckpointError(f"MLX checkpoints are .safetensors files, got {path}")
    found = {name: (tuple(a.shape), mlx_dtype_name(a.dtype)) for name, a in weights.items()}
    check_tensors(found, expected_shapes(config), "weights")
    metadata = {
        "format": FORMAT_NAME,
        "format_version": str(FORMAT_VERSION),
        "config": json.dumps(dataclasses.asdict(config)),
    }
    arrays = dict(weights)
    write_atomically(path, lambda tmp: mx.save_safetensors(str(tmp), arrays, metadata), overwrite)


def load_weights(path: Path) -> tuple[Config, dict[str, mx.array]]:
    """Read and validate an MLX checkpoint written by ``save_weights``."""
    try:
        loaded = mx.load(str(path), return_metadata=True)
    except (OSError, ValueError, RuntimeError) as err:
        raise CheckpointError(f"cannot read {path} as safetensors: {err}") from err
    if not isinstance(loaded, tuple):  # return_metadata=True always gives (arrays, metadata)
        raise CheckpointError(f"cannot read {path} as safetensors")
    arrays, metadata = loaded
    if metadata.get("format") != FORMAT_NAME:
        raise CheckpointError(f"{path} is not a {FORMAT_NAME} checkpoint")
    version = metadata.get("format_version", "")
    if not version.isdigit() or int(version) > FORMAT_VERSION:
        raise CheckpointError(
            f"{path} has format version {version!r}; this version of decoder_mlx reads "
            f"versions up to {FORMAT_VERSION}, upgrade decoder-from-scratch to read it"
        )
    try:
        raw_config: Any = json.loads(metadata.get("config", ""))
    except json.JSONDecodeError as err:
        raise CheckpointError(f"{path}: config metadata is not valid JSON") from err
    config = config_from_dict(raw_config)
    found = {name: (tuple(a.shape), mlx_dtype_name(a.dtype)) for name, a in arrays.items()}
    check_tensors(found, expected_shapes(config), str(path))
    return config, arrays


def save_model(model: Decoder, path: Path, overwrite: bool = False) -> None:
    save_weights(path, model.config, flat_parameters(model.parameters()), overwrite)


def load_model(path: Path, use_fast_kernels: bool = True) -> Decoder:
    """A ``Decoder`` with the weights of an MLX checkpoint (in training mode, like a new one)."""
    config, weights = load_weights(path)
    model = Decoder(config, use_fast_kernels)
    model.load_weights(list(weights.items()), strict=True)
    return model

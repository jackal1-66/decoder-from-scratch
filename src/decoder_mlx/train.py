"""Training for the MLX model: pretraining and fine-tuning (full or LoRA), as ``decoder.train``.

    python -m decoder_mlx.train pretrain --text shakespeare.txt --out runs/mlx-pretrain
    python -m decoder_mlx.train pretrain --text shakespeare.txt --out runs/mlx-pretrain --resume
    python -m decoder_mlx.train finetune --init model.safetensors --text alice.txt \\
        --gutenberg --lora-rank 8 --out runs/mlx-lora-r8

The recipe is that of ``decoder.train``, and shares its code where it is framework-free:
byte tokens, the same random training windows (drawn by a ``torch.Generator`` seeded with
``TrainConfig.seed``, so both frameworks see the same batches), the same held-out windows,
warmup plus cosine schedule, AdamW with betas (0.9, 0.95) and weight decay on matrices
only, and global gradient-norm clipping. MLX's Adam needs ``bias_correction=True`` to be
the Adam of PyTorch.

Differences:

* float32 throughout; ``decoder.train`` uses fp16 autocast on CUDA.
* At every evaluation the full training state is saved to ``out/train_state.safetensors``
  (model, optimiser moments, step, batch-sampler state); ``--resume`` continues from it
  and gives the same result as an uninterrupted run. Dropout masks need no saved state:
  the global PRNG is reseeded from (seed, step) before every step.
"""

import argparse
import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from mlx.utils import tree_unflatten
from torch import Tensor

from decoder.config import Config
from decoder.train import TrainConfig, batch, eval_batches, load_bytes, lr_at, split
from decoder_mlx.checkpoint import (
    CheckpointError,
    config_from_dict,
    load_model,
    save_model,
    write_atomically,
)
from decoder_mlx.lora import apply_lora
from decoder_mlx.model import Decoder, flat_parameters

logger = logging.getLogger(__name__)

STATE_FILE = "train_state.safetensors"
STATE_FORMAT = "decoder-from-scratch-mlx-train"
STATE_FORMAT_VERSION = 1


def _to_mlx(tokens: Tensor) -> mx.array:
    return mx.array(tokens.numpy().astype(np.int32))


def _scalar(a: mx.array) -> float:
    return float(np.array(a))


def evaluate(model: Decoder, data: Tensor, seq_len: int, max_windows: int = 200) -> float:
    """Bits per byte on the held-out windows that ``decoder.train.evaluate`` scores."""
    model.eval()
    total, count = 0.0, 0
    for x, y in eval_batches(data, seq_len, max_windows):
        logits = model(_to_mlx(x))
        total += _scalar(nn.losses.cross_entropy(logits, _to_mlx(y), reduction="sum"))
        count += y.numel()
    model.train()
    return total / count / math.log(2)


def make_optimizer(cfg: TrainConfig) -> optim.MultiOptimizer:
    """AdamW as configured in ``decoder.train``: decay matrices, not gains."""

    def adamw(weight_decay: float) -> optim.AdamW:
        return optim.AdamW(
            learning_rate=cfg.learning_rate,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=weight_decay,
            bias_correction=True,  # MLX's default is False; PyTorch always corrects
        )

    def is_matrix(path: str, weight: mx.array) -> bool:
        return weight.ndim >= 2

    return optim.MultiOptimizer([adamw(cfg.weight_decay), adamw(0.0)], [is_matrix])


def _save_state(
    path: Path, model: Decoder, optimizer: optim.Optimizer, step: int, cfg: TrainConfig,
    batch_rng: torch.Generator,
) -> None:  # fmt: skip
    arrays = {f"model.{k}": v for k, v in flat_parameters(model.parameters()).items()}
    arrays |= {f"optimizer.{k}": v for k, v in flat_parameters(optimizer.state).items()}
    arrays["batch_rng"] = mx.array(batch_rng.get_state().numpy())
    metadata = {
        "format": STATE_FORMAT,
        "format_version": str(STATE_FORMAT_VERSION),
        "step": str(step),
        "config": json.dumps(dataclasses.asdict(model.config)),
        "train_config": json.dumps(dataclasses.asdict(cfg)),
    }
    write_atomically(path, lambda tmp: mx.save_safetensors(str(tmp), arrays, metadata), True)


def _load_state(
    path: Path, model: Decoder, optimizer: optim.Optimizer, cfg: TrainConfig,
    batch_rng: torch.Generator,
) -> int:  # fmt: skip
    """Restore model, optimiser and sampler from ``path``; return the step to continue at."""
    if not path.exists():
        raise CheckpointError(f"cannot resume: {path} does not exist")
    loaded = mx.load(str(path), return_metadata=True)
    if not isinstance(loaded, tuple) or loaded[1].get("format") != STATE_FORMAT:
        raise CheckpointError(f"{path} is not a {STATE_FORMAT} file")
    arrays, metadata = loaded
    if metadata.get("format_version") != str(STATE_FORMAT_VERSION):
        raise CheckpointError(
            f"{path}: unsupported format version {metadata.get('format_version')}"
        )
    if config_from_dict(json.loads(metadata["config"])) != model.config:
        raise CheckpointError(f"{path} was written for a different model config")
    if json.loads(metadata["train_config"]) != dataclasses.asdict(cfg):
        raise CheckpointError(
            f"{path} was written with train config {metadata['train_config']}; "
            "resume with the same settings (steps, learning rate, seed, ...)"
        )
    weights = [(k.removeprefix("model."), v) for k, v in arrays.items() if k.startswith("model.")]
    model.load_weights(weights, strict=True)  # also catches a different LoRA setup
    state = [
        (k.removeprefix("optimizer."), v) for k, v in arrays.items() if k.startswith("optimizer.")
    ]
    optimizer.state = tree_unflatten(state)
    batch_rng.set_state(torch.from_numpy(np.array(arrays["batch_rng"])))
    return int(metadata["step"])


def _read_log(path: Path, before_step: int) -> list[str]:
    """Log lines of an interrupted run up to its checkpoint (evaluations at that step included)."""
    kept = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["step"] < before_step or (
            record["step"] == before_step and "train_bpb" not in record
        ):
            kept.append(line + "\n")
    return kept


def train(
    model: Decoder,
    train_data: Tensor,
    evals: dict[str, Tensor],
    cfg: TrainConfig,
    out: Path,
    resume: bool = False,
) -> dict[str, float]:
    """Train the trainable parameters; log to out/log.jsonl; return final evaluations.

    With ``resume``, continue from ``out/train_state.safetensors``, which must have been
    written with the same model config, ``cfg`` and LoRA setup.
    """
    batch_rng = torch.Generator().manual_seed(cfg.seed)
    model.train()
    optimizer = make_optimizer(cfg)
    seq_len = model.config.max_seq_len
    out.mkdir(parents=True, exist_ok=True)
    state_path, log_path = out / STATE_FILE, out / "log.jsonl"
    first = 0
    if resume:
        first = _load_state(state_path, model, optimizer, cfg, batch_rng)
        lines = _read_log(log_path, first) if log_path.exists() else []
        logger.info("resuming at step %d from %s", first, state_path)
    else:
        lines = []
    with log_path.open("w") as log:
        log.writelines(lines)

        def loss_fn(model: Decoder, x: mx.array, y: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(x), y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        t0 = time.perf_counter()
        for step in range(first, cfg.steps + 1):
            evaluated_before_checkpoint = resume and step == first
            if (
                step % cfg.eval_every == 0 or step == cfg.steps
            ) and not evaluated_before_checkpoint:
                record: dict[str, float] = {"step": step, "seconds": time.perf_counter() - t0}
                record |= {f"{n}_bpb": evaluate(model, d, seq_len) for n, d in evals.items()}
                log.write(json.dumps(record) + "\n")
                log.flush()
                logger.info(json.dumps(record))
                _save_state(state_path, model, optimizer, step, cfg, batch_rng)
            if step == cfg.steps:
                break
            mx.random.seed((cfg.seed << 32) | step)  # dropout masks depend on (seed, step) only
            optimizer.learning_rate = lr_at(step, cfg)
            x, y = batch(train_data, cfg.batch_size, seq_len, batch_rng, "cpu")
            loss, grads = loss_and_grad(model, _to_mlx(x), _to_mlx(y))
            grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            if cfg.log_every and step % cfg.log_every == 0:
                log.write(
                    json.dumps({"step": step, "train_bpb": _scalar(loss) / math.log(2)}) + "\n"
                )
    return {name: evaluate(model, data, seq_len) for name, data in evals.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["pretrain", "finetune"])
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--gutenberg", action="store_true", help="strip Project Gutenberg header/footer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, help="finetune: pretrained MLX model.safetensors")
    p.add_argument(
        "--forgetting-text", type=Path, help="finetune: pretraining text, to measure forgetting"
    )
    p.add_argument("--lora-rank", type=int, default=0, help="finetune: 0 = full fine-tuning")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help=f"continue from out/{STATE_FILE}")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mx.random.seed(args.seed)  # model initialisation
    data = load_bytes(args.text, args.gutenberg)
    if args.mode == "pretrain":
        config = Config(dropout=args.dropout)
        model = Decoder(config)
        train_data, val = split(data, 0.1)
        evals = {"val": val}
    else:
        if args.init is None:
            p.error("finetune needs --init")
        pretrained = load_model(args.init)
        config = dataclasses.replace(pretrained.config, dropout=args.dropout)
        model = Decoder(config)
        model.load_weights(list(flat_parameters(pretrained.parameters()).items()))
        train_data, held_out = split(data, 0.2)
        dev, test = split(held_out, 0.5)
        evals = {"dev": dev, "test": test}
        if args.forgetting_text:
            evals["pretrain_val"] = split(load_bytes(args.forgetting_text), 0.1)[1]
        if args.lora_rank:
            apply_lora(model, targets=("wq", "wk", "wv", "wo", "w1", "w2", "w3"),
                       rank=args.lora_rank, alpha=2 * args.lora_rank)  # fmt: skip

    cfg = TrainConfig(steps=args.steps, learning_rate=args.learning_rate, seed=args.seed)
    trainable = sum(a.size for a in flat_parameters(model.trainable_parameters()).values())
    final = train(model, train_data, evals, cfg, args.out, resume=args.resume)
    summary: dict[str, Any] = {
        "framework": "mlx", "mode": args.mode, "text": str(args.text),
        "lora_rank": args.lora_rank, "model_config": dataclasses.asdict(config),
        "train_config": dataclasses.asdict(cfg), "parameters": model.num_parameters(),
        "trainable_parameters": trainable, "train_bytes": len(train_data), "final_bpb": final,
    }  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.mode == "pretrain":
        save_model(model, args.out / "model.safetensors", overwrite=True)


if __name__ == "__main__":
    main()

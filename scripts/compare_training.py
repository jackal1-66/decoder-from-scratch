"""Pretrain the same model with decoder.train and decoder_mlx.train and compare the runs.

    python scripts/compare_training.py --text shakespeare.txt --out runs/compare

Both loops run on CPU in float32 from the same initial weights, on the same batches, with
dropout off, so that every difference comes from floating-point arithmetic. A control run
trains the PyTorch model again from weights perturbed by one part in 1e7, about one
float32 rounding error: if PyTorch vs perturbed PyTorch drifts apart like PyTorch vs MLX,
the drift measures how training amplifies rounding differences, not a disagreement
between the implementations.

Writes out/{torch,torch_perturbed,mlx}/log.jsonl and out/comparison.json with, for each
of the two pairs: the per-step relative difference of the training loss, the first step
where it exceeds 1e-5, 1e-4, 1e-3 and 1e-2, the relative distance between the final
weights, and each run's held-out bits per byte and time per step.
"""

import argparse
import dataclasses
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from decoder.config import Config
from decoder.model import Decoder
from decoder.train import TrainConfig, load_bytes, split
from decoder.train import train as torch_train
from decoder_mlx.convert import mlx_to_torch, torch_to_mlx
from decoder_mlx.model import Decoder as MLXDecoder
from decoder_mlx.model import flat_parameters
from decoder_mlx.train import train as mlx_train

THRESHOLDS = (1e-5, 1e-4, 1e-3, 1e-2)


def curve(out: Path) -> dict[int, float]:
    lines = (out / "log.jsonl").read_text().splitlines()
    return {r["step"]: r["train_bpb"] for r in map(json.loads, lines) if "train_bpb" in r}


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    steps = sorted(a["curve"])
    rel = [abs(a["curve"][s] - b["curve"][s]) / a["curve"][s] for s in steps]
    first_above = {
        f"{t:g}": next((s for s, r in zip(steps, rel, strict=True) if r > t), None)
        for t in THRESHOLDS
    }
    ta, tb = a["weights"], b["weights"]
    distance = math.sqrt(sum(float(((ta[n] - tb[n]) ** 2).sum()) for n in ta))
    norm = math.sqrt(sum(float((ta[n] ** 2).sum()) for n in ta))
    return {
        "loss_relative_difference": {
            "first_step_above": first_above,
            "max_first_10_steps": max(rel[:10]),
            "max_first_100_steps": max(rel[:100]),
            "max_all_steps": max(rel),
            "at_last_step": rel[-1],
        },
        "final_weights_relative_distance": distance / norm,
        "final_val_bpb": {a["name"]: a["val_bpb"], b["name"]: b["val_bpb"]},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=384)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = Config(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    )
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        log_every=1,
        seed=args.seed,
    )
    train_data, val = split(load_bytes(args.text), 0.1)
    torch.manual_seed(args.seed)
    initial = Decoder(config).state_dict()
    generator = torch.Generator().manual_seed(args.seed + 1)
    perturbed = {
        n: t * (1 + 1e-7 * torch.randn(t.shape, generator=generator)) for n, t in initial.items()
    }
    perturbed["head.weight"] = perturbed["embed.weight"]  # keep the tie

    runs: dict[str, dict[str, Any]] = {}
    for name, state in (("torch", initial), ("torch_perturbed", perturbed)):
        model = Decoder(config)
        model.load_state_dict(state)
        t0 = time.perf_counter()
        final = torch_train(model, train_data, {"val": val}, cfg, args.out / name, device="cpu")
        weights = {n: t.detach().double().numpy() for n, t in model.state_dict().items()}
        runs[name] = {
            "seconds": time.perf_counter() - t0,
            "val_bpb": final["val"],
            "weights": weights,
        }
    model_mlx = MLXDecoder(config)
    model_mlx.load_weights(list(torch_to_mlx(initial, config).items()))
    t0 = time.perf_counter()
    final = mlx_train(model_mlx, train_data, {"val": val}, cfg, args.out / "mlx")
    state = mlx_to_torch(flat_parameters(model_mlx.parameters()), config)
    runs["mlx"] = {
        "seconds": time.perf_counter() - t0,
        "val_bpb": final["val"],
        "weights": {n: t.double().numpy() for n, t in state.items()},
    }
    for name, run in runs.items():
        run["name"], run["curve"] = name, curve(args.out / name)

    report = {
        "config": dataclasses.asdict(config),
        "train_config": dataclasses.asdict(cfg),
        "parameters": Decoder(config).num_parameters(),
        "versions": {"torch": torch.__version__, "mlx": version("mlx"), "numpy": np.__version__},
        "machine": platform.machine(),
        "mlx_device": str(mx.default_device()),
        "torch_threads": torch.get_num_threads(),
        "seconds_per_step": {n: r["seconds"] / args.steps for n, r in runs.items()},
        "torch_vs_mlx": compare(runs["torch"], runs["mlx"]),
        "torch_vs_torch_perturbed": compare(runs["torch"], runs["torch_perturbed"]),
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

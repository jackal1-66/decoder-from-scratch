"""LoRA vs full fine-tuning on a new domain, and how much each forgets the old one.

Pretrained on Shakespeare, fine-tuned on Alice's Adventures in Wonderland. Each method
gets its own learning-rate sweep; the rate is chosen on Alice-dev, and Alice-test and
Shakespeare-val are reported for the chosen run. Then a LoRA rank ablation at LoRA's
chosen rate.

Finally, the key configurations are repeated with three seeds (batch order and LoRA
initialisation change; the pretrained model does not) to see what differences are noise.

Usage: python scripts/finetune_experiment.py <data dir> <runs dir>
"""

import json
import subprocess
import sys
from pathlib import Path

STEPS = 200
SWEEPS = {0: [1e-4, 3e-4, 1e-3], 8: [1e-3, 3e-3, 1e-2]}  # lora rank (0 = full) -> learning rates
RANKS = [1, 4, 8, 32]
SEEDS = [0, 1, 2]


def run(data: Path, runs: Path, rank: int, lr: float, seed: int = 0) -> dict[str, object]:
    out = (
        runs / f"finetune-r{rank}-lr{lr:g}"
        if seed == 0
        else runs / f"finetune-r{rank}-lr{lr:g}-s{seed}"
    )
    if not (out / "summary.json").exists():
        subprocess.run(
            [sys.executable, "-m", "decoder.train", "finetune",
             "--init", str(runs / "pretrain" / "model.pt"),
             "--text", str(data / "alice.txt"), "--gutenberg",
             "--forgetting-text", str(data / "shakespeare.txt"),
             "--lora-rank", str(rank), "--learning-rate", str(lr), "--steps", str(STEPS),
             "--seed", str(seed), "--out", str(out)],
            check=True, capture_output=True,
        )  # fmt: skip
    summary: dict[str, object] = json.loads((out / "summary.json").read_text())
    first = json.loads((out / "log.jsonl").read_text().splitlines()[0])  # step 0 = before
    summary["before_bpb"] = {k[: -len("_bpb")]: v for k, v in first.items() if k.endswith("_bpb")}
    return summary


def main() -> None:
    data, runs = Path(sys.argv[1]), Path(sys.argv[2])
    print(
        "| method | learning rate | trainable params | Alice dev | Alice test | Shakespeare val |"
    )
    print("|---|---|---|---|---|---|")
    best: dict[int, tuple[float, dict[str, object]]] = {}
    before = None
    for rank, lrs in SWEEPS.items():
        for lr in lrs:
            s = run(data, runs, rank, lr)
            before = s["before_bpb"]
            final = s["final_bpb"]
            assert isinstance(final, dict)
            name = "full" if rank == 0 else f"LoRA r={rank}"
            print(
                f"| {name} | {lr:g} | {s['trainable_parameters']:,} | {final['dev']:.3f}"
                f" | {final['test']:.3f} | {final['pretrain_val']:.3f} |",
                flush=True,
            )
            if rank not in best or final["dev"] < best[rank][1]["final_bpb"]["dev"]:  # type: ignore[index]
                best[rank] = (lr, s)
    assert isinstance(before, dict)
    print(
        f"\nBefore fine-tuning: Alice dev {before['dev']:.3f}, test {before['test']:.3f},"
        f" Shakespeare val {before['pretrain_val']:.3f}"
    )

    lora_lr = best[8][0]
    print(f"\nLoRA rank ablation at learning rate {lora_lr:g}:\n")
    print("| rank | trainable params | Alice test | Shakespeare val |")
    print("|---|---|---|---|")
    for rank in RANKS:
        s = run(data, runs, rank, lora_lr)
        final = s["final_bpb"]
        assert isinstance(final, dict)
        print(
            f"| {rank} | {s['trainable_parameters']:,} | {final['test']:.3f}"
            f" | {final['pretrain_val']:.3f} |",
            flush=True,
        )

    print("\nSeed repeats (mean, min-max over seeds 0, 1, 2):\n")
    print("| configuration | Alice test | Shakespeare val |")
    print("|---|---|---|")
    for rank, lr in [(0, best[0][0]), (8, lora_lr), (8, 1e-3)]:
        finals = [run(data, runs, rank, lr, seed)["final_bpb"] for seed in SEEDS]
        cells = []
        for key in ("test", "pretrain_val"):
            values = [f[key] for f in finals]  # type: ignore[index]
            cells.append(f"{sum(values) / len(values):.3f} ({min(values):.3f}-{max(values):.3f})")
        name = "full" if rank == 0 else f"LoRA r={rank}"
        print(f"| {name}, lr {lr:g} | {cells[0]} | {cells[1]} |", flush=True)


if __name__ == "__main__":
    main()

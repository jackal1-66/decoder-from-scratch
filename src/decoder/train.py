"""Training: byte-level language modelling, pretraining and fine-tuning (full or LoRA).

    python -m decoder.train pretrain --text shakespeare.txt --out runs/pretrain
    python -m decoder.train finetune --init runs/pretrain/model.pt --text alice.txt \\
        --gutenberg --lora-rank 8 --out runs/lora-r8 --forgetting-text shakespeare.txt

Tokens are bytes (vocabulary 256): no tokenizer to train, at the price of longer
sequences. Losses are reported in bits per byte (nats / ln 2), the usual unit for
byte-level models: an ideal model of English text sits somewhere near 1.

Splits are contiguous: the last ``val_fraction`` of a text is held out (random windows
would leak, because overlapping windows share most of their bytes). For fine-tuning the
held-out tail is split once more into dev (choose settings) and test (report).
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from decoder.lora import apply_lora
from decoder.model import Config, Decoder

logger = logging.getLogger(__name__)

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def load_bytes(path: Path, gutenberg: bool = False) -> Tensor:
    text = path.read_text(encoding="utf-8")
    if gutenberg:
        # Keep only the book: drop Project Gutenberg's licence header and footer.
        text = text.split(GUTENBERG_START, 1)[1].split("\n", 1)[1]
        text = text.split(GUTENBERG_END, 1)[0]
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def split(data: Tensor, held_out: float) -> tuple[Tensor, Tensor]:
    cut = int(len(data) * (1 - held_out))
    return data[:cut], data[cut:]


def batch(
    data: Tensor, size: int, seq_len: int, gen: torch.Generator, device: str
) -> tuple[Tensor, Tensor]:
    """Random windows; targets are the inputs shifted by one byte."""
    starts = torch.randint(0, len(data) - seq_len - 1, (size,), generator=gen)
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def eval_batches(
    data: Tensor, seq_len: int, max_windows: int = 200, batch_size: int = 32
) -> Iterator[tuple[Tensor, Tensor]]:
    """Inputs and targets of the first non-overlapping windows of held-out data, on CPU."""
    n = min(max_windows, (len(data) - 1) // seq_len)
    for i in range(0, n, batch_size):
        starts = range(i * seq_len, min(i + batch_size, n) * seq_len, seq_len)
        x = torch.stack([data[s : s + seq_len] for s in starts])
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        yield x, y


@torch.no_grad()
def evaluate(
    model: Decoder, data: Tensor, seq_len: int, device: str, max_windows: int = 200
) -> float:
    """Bits per byte over non-overlapping windows of held-out data (deterministic)."""
    model.eval()
    total, count = 0.0, 0
    for x, y in eval_batches(data, seq_len, max_windows):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x)
        total += F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
        ).item()
        count += y.numel()
    model.train()
    return total / count / math.log(2)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_lr_fraction: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    log_every: int = 50  # training loss of every n-th batch; 0 = never
    seed: int = 0


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * cosine)


def train(
    model: Decoder,
    train_data: Tensor,
    evals: dict[str, Tensor],
    cfg: TrainConfig,
    out: Path,
    device: str = "cuda",
) -> dict[str, float]:
    """Train the parameters that require grad; log to out/log.jsonl; return final evals."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    # Weight decay on matrices only: decaying gains (RMSNorm) and biases toward zero
    # just fights the normalisation without regularising anything useful.
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))
    scaler = GradScaler(device, enabled=device == "cuda")
    seq_len = model.config.max_seq_len
    out.mkdir(parents=True, exist_ok=True)
    with (out / "log.jsonl").open("w") as log:
        t0 = time.perf_counter()
        for step in range(cfg.steps + 1):
            if step % cfg.eval_every == 0 or step == cfg.steps:
                record: dict[str, float] = {"step": step, "seconds": time.perf_counter() - t0}
                record.update(
                    {
                        f"{name}_bpb": evaluate(model, data, seq_len, device)
                        for name, data in evals.items()
                    }
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                logger.info(json.dumps(record))
            if step == cfg.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, cfg)
            x, y = batch(train_data, cfg.batch_size, seq_len, gen, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            if cfg.log_every and step % cfg.log_every == 0:
                log.write(json.dumps({"step": step, "train_bpb": loss.item() / math.log(2)}) + "\n")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
    final = {name: evaluate(model, data, seq_len, device) for name, data in evals.items()}
    return final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["pretrain", "finetune"])
    p.add_argument("--text", type=Path, required=True)
    p.add_argument("--gutenberg", action="store_true", help="strip Project Gutenberg header/footer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, help="finetune: pretrained model.pt")
    p.add_argument(
        "--forgetting-text", type=Path, help="finetune: pretraining text, to measure forgetting"
    )
    p.add_argument("--lora-rank", type=int, default=0, help="finetune: 0 = full fine-tuning")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = load_bytes(args.text, args.gutenberg)
    if args.mode == "pretrain":
        config = Config(dropout=args.dropout)
        model = Decoder(config)
        train_data, val = split(data, 0.1)
        evals = {"val": val}
    else:
        checkpoint = torch.load(args.init, map_location="cpu")
        config = Config(**{**checkpoint["config"], "dropout": args.dropout})
        model = Decoder(config)
        model.load_state_dict(checkpoint["model"])
        train_data, held_out = split(data, 0.2)
        dev, test = split(held_out, 0.5)
        evals = {"dev": dev, "test": test}
        if args.forgetting_text:
            evals["pretrain_val"] = split(load_bytes(args.forgetting_text), 0.1)[1]
        if args.lora_rank:
            apply_lora(model, targets=("wq", "wk", "wv", "wo", "w1", "w2", "w3"),
                       rank=args.lora_rank, alpha=2 * args.lora_rank)  # fmt: skip

    cfg = TrainConfig(steps=args.steps, learning_rate=args.learning_rate, seed=args.seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final = train(model, train_data, evals, cfg, args.out)
    summary = {
        "mode": args.mode, "text": str(args.text), "lora_rank": args.lora_rank,
        "model_config": asdict(config), "train_config": asdict(cfg),
        "parameters": model.num_parameters(), "trainable_parameters": trainable,
        "train_bytes": len(train_data), "final_bpb": final,
    }  # fmt: skip
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.mode == "pretrain":
        torch.save({"config": asdict(config), "model": model.state_dict()}, args.out / "model.pt")


if __name__ == "__main__":
    main()

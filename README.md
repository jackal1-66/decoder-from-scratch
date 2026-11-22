# decoder-from-scratch
Developed with the assistance of AI coding tools.

A decoder-only transformer written from scratch in PyTorch, covering the mechanisms that
matter in practice: causal attention with rotary embeddings and grouped-query heads,
RMSNorm and SwiGLU, a KV cache, temperature/top-k/top-p sampling, and a LoRA layer with
merge/unmerge. Every mechanism has a test that checks the *property* it is supposed to
have (causality, cache equivalence, relative-position invariance, identity at LoRA
initialisation), not just shapes. Three small experiments on one V100 measure the claims
people usually make about them.

| File | Content |
|---|---|
| `src/decoder/model.py` | RMSNorm, RoPE, causal GQA attention, SwiGLU, pre-norm blocks, tied head, KV cache |
| `src/decoder/sampling.py` | greedy / temperature / top-k / top-p, generation with or without the cache |
| `src/decoder/lora.py` | `LoRALinear` (merge, unmerge) and `apply_lora` |
| `src/decoder/train.py` | byte-level pretraining and fine-tuning (full or LoRA), fp16 on Volta |
| `scripts/` | generation benchmark; LoRA vs full fine-tuning experiment |

## Results (one Tesla V100, all from the scripts in `scripts/`)

- **Pretraining**, 9.5M parameters on Tiny Shakespeare: 1.972 bits/byte on held-out text
  after 1,000 steps (82 s). A 3,000-step run overfits after step 1,000, reaching 2.514.
- **KV cache:** no speed-up for the small model (each step is about 6 ms of launch overhead
  whatever its length). 5.0× for a 136M-parameter model at batch 8 and 1,024 new tokens,
  where recomputing the prefix dominates.
- **LoRA vs full fine-tuning** (Shakespeare → Alice, learning rate chosen on dev, 3 seeds):
  full fine-tuning reached 1.821 bits/byte on Alice-test and forgot less (Shakespeare
  2.461) than LoRA r=8 (1.919 / 2.884). LoRA at a 10× lower rate forgot least (2.378) but
  learned least (2.008). Higher LoRA rank forgot monotonically more.

The numbers come from `scripts/benchmark_generation.py`, `scripts/finetune_experiment.py`
and the training logs written by `python -m decoder.train`.

## Data

- Tiny Shakespeare (`karpathy/char-rnn`, commit `6f9487a6`), the text is public domain;
  sha256 `86c4e6aa…65ed`.
- *Alice's Adventures in Wonderland*, Project Gutenberg eBook #11 (public domain in the
  US); the Gutenberg licence header and footer are stripped before use; sha256 of the
  downloaded file `01b38ea4…725e`.

## Reproducing

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
ruff check . && ruff format --check . && mypy && pytest

python -m decoder.train pretrain --text data/shakespeare.txt --out runs/pretrain --steps 1000
python scripts/benchmark_generation.py runs/pretrain/model.pt
python scripts/finetune_experiment.py data runs
```

## Licence

MIT.

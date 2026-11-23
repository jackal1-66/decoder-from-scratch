# decoder-from-scratch
Developed with the assistance of AI coding tools.

A decoder-only transformer written from scratch in PyTorch, covering the mechanisms that
matter in practice: causal attention with rotary embeddings and grouped-query heads,
RMSNorm and SwiGLU, a KV cache, temperature/top-k/top-p sampling, and a LoRA layer with
merge/unmerge. Every mechanism has a test that checks the *property* it is supposed to
have (causality, cache equivalence, relative-position invariance, identity at LoRA
initialisation), not just shapes. Three small experiments on one V100 measure the claims
people usually make about them.

An MLX implementation (`decoder_mlx`) computes the same function: same configuration, same
parameter names, checkpoints that convert bit for bit, and tests that the two agree
to float32 precision, in inference and in training.

| File | Content |
|---|---|
| `src/decoder/config.py` | `Config`, shared by both implementations |
| `src/decoder/model.py` | RMSNorm, RoPE, causal GQA attention, SwiGLU, pre-norm blocks, tied head, KV cache |
| `src/decoder/sampling.py` | greedy / temperature / top-k / top-p, generation with or without the cache |
| `src/decoder/lora.py` | `LoRALinear` (merge, unmerge) and `apply_lora` |
| `src/decoder/train.py` | byte-level pretraining and fine-tuning (full or LoRA), fp16 on Volta |
| `src/decoder_mlx/` | the same five modules in MLX, plus `checkpoint.py`, `convert.py` (`decoder-convert`) and `benchmark.py` (`decoder-benchmark`) |
| `scripts/` | generation benchmark; LoRA vs full fine-tuning experiment; PyTorch/MLX equivalence and training comparison |

## Installation

```bash
pip install decoder-from-scratch            # PyTorch only
pip install "decoder-from-scratch[mlx]"     # adds MLX
```

Python 3.11+ (CI runs 3.12), torch 2.4+; the `mlx` extra needs MLX 0.32.4+ and numpy 2.0+
(CI tests these minimum versions of torch, numpy and MLX). The PyTorch package does not import MLX; importing
`decoder_mlx` without it raises an `ImportError` that says what to install.

On macOS, MLX runs on the Apple-silicon GPU (Metal). On Linux the extra installs MLX's CPU
backend, which compiles some kernels at run time and therefore needs a C++ compiler
(`g++`) on the machine. MLX also has a CUDA backend (`pip install "mlx[cuda12]"`), but its
wheels contain no kernels for older GPUs: on a Tesla V100 (compute capability 7.0) the
first kernel launch fails with "no kernel image is available for execution on the device".

For development, install torch from the index that matches your hardware first, so that
pip cannot replace it with a build for another CUDA version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu
pip install -e ".[dev,mlx]"
ruff check . && ruff format --check . && mypy && pytest
```

## Usage

PyTorch training, as before (`--help` lists the options):

```bash
python -m decoder.train pretrain --text data/shakespeare.txt --out runs/pretrain --steps 1000
```

Convert a checkpoint to MLX and back. Config, tensor names, shapes and dtypes are checked
before anything is written; an error lists every problem and leaves no partial file:

```bash
decoder-convert torch-to-mlx runs/pretrain/model.pt runs/pretrain/model.safetensors
decoder-convert mlx-to-torch runs/pretrain/model.safetensors model.pt [--overwrite]
```

Generate with MLX. Sampling with `temperature > 0` takes an explicit PRNG key, and the same
key gives the same text:

```python
from pathlib import Path

import mlx.core as mx

from decoder_mlx.checkpoint import load_model
from decoder_mlx.sampling import generate

model = load_model(Path("runs/pretrain/model.safetensors"))
model.eval()
prompt = mx.array([list(b"ROMEO:\n")], dtype=mx.int32)
tokens = generate(
    model, prompt, max_new_tokens=200, temperature=0.8, top_k=40, key=mx.random.key(0)
)
print(bytes(tokens[0].tolist()).decode("utf-8", errors="replace"))
```

LoRA in MLX: `apply_lora` freezes everything else, `nn.value_and_grad` then differentiates
the adapters only, and `merge` folds them into the base weights:

```python
import mlx.nn as nn
import mlx.optimizers as optim

from decoder_mlx.lora import LoRALinear, apply_lora

apply_lora(model, rank=8, alpha=16.0)
optimizer = optim.AdamW(learning_rate=1e-3, bias_correction=True)


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


x = mx.array([list(b"To be, or not to be")], dtype=mx.int32)
loss, grads = nn.value_and_grad(model, loss_fn)(model, x[:, :-1], x[:, 1:])
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)

for _, module in model.named_modules():
    if isinstance(module, LoRALinear):
        module.merge()  # inference now costs what the base model costs
```

Train in MLX with the recipe of `decoder.train` (same batches, schedule and optimiser).
The full training state is saved at every evaluation, and `--resume` continues exactly
where an interrupted run stopped:

```bash
python -m decoder_mlx.train pretrain --text data/shakespeare.txt --out runs/mlx-pretrain
python -m decoder_mlx.train pretrain --text data/shakespeare.txt --out runs/mlx-pretrain --resume
python -m decoder_mlx.train finetune --init runs/mlx-pretrain/model.safetensors \
    --text data/alice.txt --gutenberg --lora-rank 8 --out runs/mlx-lora-r8
```

Benchmark both frameworks on the current machine (`--config` takes a JSON `Config`):

```bash
decoder-benchmark --out benchmark.json --markdown benchmark.md
```

## Results (one Tesla V100, all from the scripts in `scripts/`)

- **Pretraining**, 9.5M parameters on Tiny Shakespeare: 1.972 bits/byte on held-out text
  after 1,000 steps (82 s). A 3,000-step run overfits after step 1,000, reaching 2.514.
  Rerun with torch 2.14.1: 1.982 (81 s); fp16 kernels on the GPU are not bitwise
  deterministic across versions, and the run seeds the batches but not the initial weights.
- **KV cache:** no speed-up for the small model (each step is about 6 ms of launch overhead
  whatever its length). 5.0× for a 136M-parameter model at batch 8 and 1,024 new tokens,
  where recomputing the prefix dominates.
- **LoRA vs full fine-tuning** (Shakespeare → Alice, learning rate chosen on dev, 3 seeds):
  full fine-tuning reached 1.821 bits/byte on Alice-test and forgot less (Shakespeare
  2.461) than LoRA r=8 (1.919 / 2.884). LoRA at a 10× lower rate forgot least (2.378) but
  learned least (2.008). Higher LoRA rank forgot monotonically more.

The numbers come from `scripts/benchmark_generation.py`, `scripts/finetune_experiment.py`
and the training logs written by `python -m decoder.train`.

## PyTorch and MLX compute the same function

Measured on CPU in float32 (MLX 0.32.4, torch 2.14.1), with `scripts/mlx_equivalence.py`:
the PyTorch weights are converted to MLX and both models score the same tokens. A float64
PyTorch run is the reference, so each framework's own float32 rounding error can be
compared with the difference between them.

| Max logit difference (9.5M parameters, 8 × 256 tokens) | random weights | trained weights, held-out text |
|---|---|---|
| PyTorch fp32 vs fp64 | 3.1e-6 | 2.0e-5 |
| MLX fp32 vs fp64 | 2.9e-6 | 2.8e-5 |
| **PyTorch fp32 vs MLX fp32** | **2.9e-6** | **2.6e-5** (1.3e-6 of the largest logit) |
| greedy generation, 8 × 100 tokens | identical | identical |

The frameworks differ from each other by as much as each differs from the exact result:
about 10 float32 units in the last place. The tests therefore require agreement within
1e-5 of the largest logit, which is 5× the largest difference measured on the test
configuration, while the wrong RoPE layout misses it 100,000-fold. They also check that
cached and uncached generation agree, that LoRA is the identity at initialisation and
agrees across frameworks merged and unmerged, and that converting PyTorch → MLX → PyTorch
reproduces every tensor bit for bit in float32, float16 and bfloat16.

Implementation conventions that matter for this:

- **RoPE layout.** Both models rotate the dimension pairs (i, i + D/2) ("half-split").
  That is `mx.fast.rope(..., traditional=False)`; `traditional=True` rotates adjacent pairs
  (2i, 2i + 1) and gives a different model without any error.
- **Causal mask with a cache.** With T new queries and S cached plus new keys, query i sees
  keys up to S − T + i, the bottom-right alignment of `mask="causal"` in
  `mx.fast.scaled_dot_product_attention`.
- **Fast kernels.** `decoder_mlx` implements RoPE and attention explicitly, as the PyTorch
  model does, and through `mx.fast` (the default, `use_fast_kernels=True`); the tests check
  that both paths agree. The fused attention has no dropout, so training with attention
  dropout uses the explicit path.
- **Tied head.** MLX has no shared parameters; the head is `embed.as_linear`. The converter
  checks that the PyTorch head equals the embedding and restores the sharing on the way back.
- **Adam.** `mlx.optimizers.Adam(W)` defaults to `bias_correction=False`; PyTorch always
  corrects. `decoder_mlx.train` sets it, which makes the two optimisers the same formula.

### Training

`scripts/compare_training.py` trains the same model in both frameworks from the same
initial weights on the same batches: 1,000 steps on Tiny Shakespeare, float32 on CPU, no
dropout, the full schedule of `decoder.train` (warm-up, cosine decay, AdamW, gradient
clipping). As a control, PyTorch is also trained from initial weights multiplied by
(1 + 1e-7 · noise), a perturbation the size of float32 rounding. The model is the small
configuration (d_model 128, 4 layers, 0.82M parameters) because MLX's CPU build is slow on
this machine (see [Benchmarks](#benchmarks)): 9.6 s per step against 0.08 s for PyTorch,
and about 450 s per step for the 9.5M model.

| 1,000 steps, 0.82M parameters | PyTorch vs MLX | PyTorch vs perturbed PyTorch |
|---|---|---|
| largest relative loss difference, steps 1–10 | 1.8e-6 | 8.8e-8 |
| largest relative loss difference, all steps | 4.9e-6 | 1.4e-6 |
| relative loss difference at step 1,000 | 6.9e-7 | 1.7e-7 |
| relative distance between final weights | 4.7e-5 | 1.2e-5 |
| validation bits per byte | 2.199348 vs 2.199344 | 2.199348 vs 2.199348 |

The two curves never differ by more than 1e-5 of the loss. The difference starts at the
size of the forward-pass difference in the table above and does not grow with training:
it is smallest at the last step. Training amplifies rounding differences, as the control
shows (a 1e-7 change to the weights becomes a 1.2e-5 change after 1,000 steps), and MLX
starts from larger ones, since each step's gradients already differ by float32 rounding;
the outcome is the same model to within 4e-6 bits per byte. The tests run the same
comparison for 20 steps, with a 1e-5 tolerance.

The bits-per-byte figure is not comparable with the 9.5M-parameter result above
(1.972): this model is about 12 times smaller.

## Benchmarks

`decoder-benchmark`, run twice on this machine: once with PyTorch on the GPU (the default)
and once with `--torch-device cpu`. Same weights, float32, the 9.5M-parameter model;
median of 5 runs after 2 warm-up runs, with every interquartile range below 2% of the
median. Xeon Gold 6132 (28 cores), Tesla V100, PyTorch 2.14.1, MLX 0.32.4 on CPU.

| median time (batch 8) | PyTorch, V100 | PyTorch, CPU | MLX, CPU | MLX compiled, CPU |
|---|---:|---:|---:|---:|
| forward, 256 tokens | 8.0 ms | 149 ms | 22.7 s | 22.7 s |
| generate 64 tokens, KV cache | 371 ms | 505 ms | 5.7 s | – |
| generate 64 tokens, no cache | 387 ms | 1,048 ms | 172 s | – |

(MLX CPU times are from the `--torch-device cpu` run; the other run measured them within 4%.)

These numbers say nothing about MLX on a Mac:

- **This is MLX's slowest configuration.** On Linux, `mlx[cpu]` links the reference BLAS
  that ships in the wheel (`ldd mlx/lib/libmlx.so` lists `libblas.so.3`), not an optimised
  one, and matrix products dominate the time. The MLX times are proportional to the number
  of token positions computed: a forward pass computes 8 × 256 = 2,048, cached generation
  8 × 64 = 512 (a quarter of the time) and uncached generation 8 × (1 + … + 64) = 16,640
  (7.6 times the time). For the same reason `mx.compile`, which fuses element-wise
  operations, changes nothing here. On Apple silicon, MLX runs on the GPU through Metal.
- **GPU against CPU compares hardware, not frameworks.** The like-for-like comparison is
  the two CPU columns.
- **What carries over is the effect of the KV cache.** It reduces MLX's generation time 30×
  and PyTorch's 2× on CPU, where the cost follows the arithmetic. On the V100 it saves only
  4%: at batch 8 every decoding step is small, and both versions spend about 6 ms per token
  mostly on starting the work rather than on arithmetic.
- **Generation is not compiled.** Each decoding step has a new key length, and a new
  shape makes `mx.compile` trace and compile again.

## How MLX differs from PyTorch, and why it matters on unified memory

- **One memory, no devices.** On Apple silicon, CPU and GPU share the same physical memory.
  MLX arrays have no device: the device is chosen per operation, and the same array can feed
  a CPU operation and then a GPU one with no copy. PyTorch code moves tensors with
  `.to(device)`, which on a discrete GPU copies over PCIe and, on a Mac with the MPS
  backend, still goes through a separate allocation. Data loading, tokenisation and
  sampling can stay on the CPU in MLX without transfers, and a model can use most of the
  machine's RAM instead of a separate, smaller GPU memory, which decides how large a model
  a laptop can run.
- **Lazy evaluation.** Operations build a graph; nothing runs until a result is needed or
  `mx.eval` is called. This lets MLX schedule and fuse work and skip what is never used,
  but it changes how code is written and measured: benchmarks must evaluate inside the timed
  region, training loops must evaluate parameters and optimiser state every step or the
  graph keeps growing, and errors surface at evaluation rather than at the line that
  caused them.
- **Function transformations instead of a tape.** `mx.grad` and `nn.value_and_grad` return
  functions that compute gradients; there is no `.grad` attribute and no `zero_grad`.
  Parameters, gradients and optimiser state are nested dictionaries of the same shape.
- **Immutable arrays.** An update creates a new array (MLX reuses the memory when it can), so
  snapshots need no copy; `KVCache` slice assignment records an update rather than
  writing into a buffer.
- **Explicit randomness.** Random functions take PRNG keys, so results do not depend on the
  order of calls. The global generator used by `nn.Dropout` cannot be saved, so the
  training loop reseeds it from (seed, step) to make resume exact.
- **Compilation.** `mx.compile` fuses element-wise operations; it recompiles for every new
  input shape, and arrays reached through a closure are frozen as constants unless they are
  passed as `inputs`, for example `inputs=model.state`.

## Data

- Tiny Shakespeare (`karpathy/char-rnn`, commit `6f9487a6`), the text is public domain;
  sha256 `86c4e6aa…65ed`.
- *Alice's Adventures in Wonderland*, Project Gutenberg eBook #11 (public domain in the
  US); the Gutenberg licence header and footer are stripped before use; sha256 of the
  downloaded file `01b38ea4…725e`.

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,mlx]"
ruff check . && ruff format --check . && mypy && pytest

python -m decoder.train pretrain --text data/shakespeare.txt --out runs/pretrain --steps 1000
python scripts/benchmark_generation.py runs/pretrain/model.pt
python scripts/finetune_experiment.py data runs

python scripts/mlx_equivalence.py
python scripts/mlx_equivalence.py --checkpoint runs/pretrain/model.pt --text data/shakespeare.txt
python scripts/compare_training.py --text data/shakespeare.txt --out runs/compare
decoder-benchmark --out benchmark.json --markdown benchmark.md
decoder-benchmark --out benchmark-cpu.json --markdown benchmark-cpu.md --torch-device cpu
```

## Licence

MIT.

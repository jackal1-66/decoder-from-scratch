"""The decoder of ``decoder.model`` in MLX: same architecture, ``Config`` and parameter names.

Shape names as in ``decoder.model``: B batch, T new tokens, S tokens seen so far, H query
heads, K key/value heads, D head size, C model width, V vocabulary.

Two implementations of the position-dependent parts, selected by ``use_fast_kernels``:

* explicit: RoPE with precomputed cos/sin tables and attention as masked softmax, written
  out exactly as in the PyTorch model;
* fast (the default): ``mx.fast.rope`` and ``mx.fast.scaled_dot_product_attention``.

Both give the same result (``tests/test_mlx_model.py``). Two conventions matter for that:

* RoPE layout. This model rotates the pairs (i, i + D/2) ("half-split", as GPT-NeoX and
  Llama in Hugging Face do), which is ``mx.fast.rope(..., traditional=False)``.
  ``traditional=True`` rotates the adjacent pairs (2i, 2i + 1) of the original RoPE paper
  and would silently give a different model.
* Causal mask with a cache. With T new queries and S = start + T keys, query i may see
  keys up to start + i: the mask is aligned to the bottom-right corner of the T x S score
  matrix, which is what ``mask="causal"`` does in ``mx.fast.scaled_dot_product_attention``.

The fast attention kernel has no dropout, so while training with ``dropout > 0`` attention
falls back to the explicit path.
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from decoder.config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # Computed in float32: squaring fp16 activations overflows easily.
        x32 = x.astype(mx.float32)
        normed = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return normed.astype(x.dtype) * self.weight


def rope_tables(head_dim: int, max_seq_len: int, base: float) -> tuple[mx.array, mx.array]:
    """cos and sin of the rotation angles, shape (max_seq_len, head_dim).

    Dimension pair i rotates at frequency ``base^(-2i/head_dim)``; see ``decoder.model``.
    """
    inv_freq = base ** (-mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    angles = mx.arange(max_seq_len, dtype=mx.float32)[:, None] * inv_freq[None, :]
    angles = mx.concatenate([angles, angles], axis=-1)  # pairs are (i, i + head_dim/2)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate each pair (i, i + D/2) of ``x`` (B, heads, T, D) by its position's angle."""
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class KVCache:
    """Keys and values of every layer for all positions processed so far.

    The buffers are preallocated to ``max_seq_len`` as in the PyTorch cache. MLX arrays
    are immutable values: slice assignment rebinds ``self.k[layer]`` to a new array in the
    lazy graph rather than writing into the old buffer, and MLX reuses the buffer when
    the old array has no other reference.
    """

    def __init__(self, config: Config, batch: int, dtype: mx.Dtype = mx.float32) -> None:
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        self.batch = batch
        shape = (batch, config.n_kv_heads, config.max_seq_len, config.head_dim)
        self.k = [mx.zeros(shape, dtype=dtype) for _ in range(config.n_layers)]
        self.v = [mx.zeros(shape, dtype=dtype) for _ in range(config.n_layers)]

    def update(self, layer: int, start: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Store k, v (B, K, T, D) at positions start..start+T; return all S = start+T."""
        end = start + k.shape[2]
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    def __init__(self, config: Config, layer: int, use_fast_kernels: bool) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads, self.n_kv_heads, self.head_dim = (
            config.n_heads, config.n_kv_heads, config.head_dim,
        )  # fmt: skip
        self.rope_base = config.rope_base
        self.attn_dropout = config.dropout
        self.use_fast_kernels = use_fast_kernels
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, start: int, cache: KVCache | None
    ) -> mx.array:
        B, T, _ = x.shape
        H, K, D = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = self.wk(x).reshape(B, T, K, D).transpose(0, 2, 1, 3)  # (B, K, T, D)
        v = self.wv(x).reshape(B, T, K, D).transpose(0, 2, 1, 3)

        if self.use_fast_kernels:
            q, k = self._fast_rope(q, start), self._fast_rope(k, start)
        else:
            positions = slice(start, start + T)
            q, k = (
                apply_rope(q, cos[positions], sin[positions]),
                apply_rope(k, cos[positions], sin[positions]),
            )
        if cache is not None:
            k, v = cache.update(self.layer, start, k, v)  # now (B, K, S, D)

        if self.use_fast_kernels and not (self.training and self.attn_dropout > 0):
            # Handles GQA (K < H) itself, without materialising repeated keys and values.
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1 / math.sqrt(D), mask="causal"
            )
        else:
            out = self._explicit_attention(q, k, v, start)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, T, H * D))

    def _fast_rope(self, x: mx.array, start: int) -> mx.array:
        # traditional=False: half-split pairs (i, i + D/2), as apply_rope.
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=start
        )

    def _explicit_attention(self, q: mx.array, k: mx.array, v: mx.array, start: int) -> mx.array:
        T, S = q.shape[2], k.shape[2]
        # GQA: each group of H/K query heads reads the same key/value head.
        group = self.n_heads // self.n_kv_heads
        k = mx.repeat(k, group, axis=1)  # (B, H, S, D)
        v = mx.repeat(v, group, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)  # (B, H, T, S)
        # Query i sits at absolute position start + i and may see keys j <= start + i.
        query_pos = mx.arange(start, start + T)[:, None]
        key_pos = mx.arange(S)[None, :]
        scores = mx.where(key_pos > query_pos, -mx.inf, scores)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out: mx.array = self.dropout(weights) @ v  # (B, H, T, D)
        return out


class SwiGLU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)  # value
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)  # back to the stream
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        out: mx.array = self.dropout(self.w2(nn.silu(self.w1(x)) * self.w3(x)))
        return out


class Block(nn.Module):
    def __init__(self, config: Config, layer: int, use_fast_kernels: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config, layer, use_fast_kernels)
        self.ff_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ff = SwiGLU(config)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, start: int, cache: KVCache | None
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, start, cache)
        return x + self.ff(self.ff_norm(x))


class Decoder(nn.Module):
    """Decoder-only transformer; see ``decoder.model.Decoder`` for the architecture.

    Parameters are named as in the PyTorch ``state_dict`` (``embed.weight``,
    ``blocks.0.attn.wq.weight``, ...). The output head is tied to the embedding and has no
    parameter of its own: logits are ``embed.as_linear(x)``. Like an ``nn.Module`` in
    PyTorch, a new model is in training mode; call ``eval()`` before inference if
    ``config.dropout > 0``.
    """

    def __init__(self, config: Config, use_fast_kernels: bool = True) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = [Block(config, i, use_fast_kernels) for i in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        # The leading underscore keeps the tables out of parameters(): not trained, not saved.
        self._cos, self._sin = rope_tables(config.head_dim, config.max_seq_len, config.rope_base)
        self._init_weights()

    def _init_weights(self) -> None:
        # As in decoder.model: N(0, 0.02), residual projections scaled down by depth.
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                module.weight = 0.02 * mx.random.normal(module.weight.shape)
        for block in self.blocks:
            for proj in (block.attn.wo, block.ff.w2):
                std = 0.02 / math.sqrt(2 * self.config.n_layers)
                proj.weight = std * mx.random.normal(proj.weight.shape)

    def __call__(self, tokens: mx.array, start: int = 0, cache: KVCache | None = None) -> mx.array:
        """Logits (B, T, V) for integer tokens (B, T) occupying positions start..start+T.

        Shapes and dtypes are checked; token values are not, because that would force an
        evaluation on every call (``decoder_mlx.sampling.generate`` checks its prompt).
        """
        if tokens.ndim != 2 or not mx.issubdtype(tokens.dtype, mx.integer):
            raise ValueError(
                f"tokens must be an integer array of shape (B, T), got {tokens.dtype} "
                f"{tokens.shape}"
            )
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if start + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"positions {start}..{start + tokens.shape[1]} exceed max_seq_len "
                f"{self.config.max_seq_len}"
            )
        if cache is not None and cache.batch != tokens.shape[0]:
            raise ValueError(f"cache batch {cache.batch} != tokens batch {tokens.shape[0]}")
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self._cos, self._sin, start, cache)
        logits: mx.array = self.embed.as_linear(self.norm(x))
        return logits

    def num_parameters(self) -> int:
        return sum(p.size for p in named_parameters(self).values())


def named_parameters(module: nn.Module) -> dict[str, mx.array]:
    """All parameters of ``module`` by dotted name, as in a PyTorch ``state_dict``."""
    flat = tree_flatten(module.parameters(), destination={})
    assert isinstance(flat, dict)  # tree_flatten returns the type of `destination`
    return flat

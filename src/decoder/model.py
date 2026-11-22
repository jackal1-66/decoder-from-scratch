"""A decoder-only transformer (the architecture family of GPT, Llama, Qwen), from scratch.

Shape names used throughout: B batch, T sequence length (new tokens in this call),
S total length seen so far (cache + new), H query heads, K key/value heads, D head size,
C model width (= H * D), V vocabulary.

Architecture, top to bottom (the "pre-norm Llama" recipe):

    tokens -> embedding
           -> N x [ x = x + Attention(RMSNorm(x));  x = x + SwiGLU(RMSNorm(x)) ]
           -> RMSNorm -> linear head (tied to the embedding) -> logits

* **Pre-norm residual blocks.** Each block adds a correction to a residual stream that
  runs straight from input to output. Normalising the *input* of each sub-layer (rather
  than its output, as the original 2017 transformer did) keeps the residual path an
  identity, so gradients reach early layers undamped and deep stacks train stably.
* **RMSNorm** rescales a vector to unit root-mean-square and multiplies by a learned
  gain. Unlike LayerNorm it does not subtract the mean; it is cheaper and works as well.
* **Causal self-attention** lets position t mix information from positions <= t only.
* **RoPE** (rotary position embedding) encodes position by rotating query and key
  vectors, so attention scores depend on *relative* position (see ``apply_rope``).
* **Grouped-query attention (GQA):** H query heads share K < H key/value heads. The KV
  cache, which dominates inference memory for long contexts, shrinks by H/K.
* **SwiGLU feed-forward:** ``W2 (silu(W1 x) * W3 x)``, a gated MLP; the gate lets the
  network multiply features, which plain ReLU MLPs cannot do in one layer.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from decoder.config import Config as Config  # re-exported: the public name predates config.py


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Computed in float32: squaring fp16 activations overflows easily.
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def rope_tables(head_dim: int, max_seq_len: int, base: float) -> tuple[Tensor, Tensor]:
    """cos and sin of the rotation angles, shape (max_seq_len, head_dim).

    Dimension pair i rotates at frequency ``base^(-2i/head_dim)``: the first pairs spin
    fast (resolving nearby positions), the last ones slowly (long-range position), like
    the hands of a clock with many hands.
    """
    inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)  # pairs are (i, i + head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each pair of dimensions of ``x`` (B, heads, T, D) by its position's angle.

    For one pair, (a, b) -> (a cos θ - b sin θ, a sin θ + b cos θ): a 2-D rotation by θ.
    A query at position m and a key at position n are rotated by mθ and nθ, so their dot
    product depends on (m - n)θ only. That is the whole point: attention scores become a
    function of relative position, with no learned position table to run out of.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class KVCache:
    """Keys and values of every layer for all positions processed so far.

    Autoregressive generation feeds one new token per step. Without a cache, every step
    recomputes keys and values for the whole prefix: O(S^2) work per token. With it, a
    step computes them for the new token only and attends to the stored ones: O(S).
    Memory: 2 (k and v) x layers x K heads x D x S per sequence, which is why GQA helps.
    """

    def __init__(self, config: Config, batch: int, device: torch.device, dtype: torch.dtype):
        shape = (batch, config.n_kv_heads, config.max_seq_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]

    def update(self, layer: int, start: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store k, v (B, K, T, D) at positions start..start+T; return all S = start+T."""
        end = start + k.shape[2]
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    def __init__(self, config: Config, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads, self.n_kv_heads, self.head_dim = (
            config.n_heads, config.n_kv_heads, config.head_dim,
        )  # fmt: skip
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, start: int, cache: KVCache | None
    ) -> Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, K, T, D)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        positions = slice(start, start + T)
        q, k = (
            apply_rope(q, cos[positions], sin[positions]),
            apply_rope(k, cos[positions], sin[positions]),
        )
        if cache is not None:
            k, v = cache.update(self.layer, start, k, v)  # now (B, K, S, D)
        S = k.shape[2]

        # GQA: each group of H/K query heads reads the same key/value head.
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)  # (B, H, S, D)
        v = v.repeat_interleave(group, dim=1)

        # Scaled dot product: dividing by sqrt(D) keeps the scores' variance ~1 for
        # unit-variance q, k, so the softmax does not saturate as D grows.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, S)
        # Query i sits at absolute position start + i and may see keys j <= start + i.
        query_pos = torch.arange(start, start + T, device=x.device)[:, None]
        key_pos = torch.arange(S, device=x.device)[None, :]
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        out = self.dropout(weights) @ v  # (B, H, T, D)
        return self.wo(out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)  # value
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)  # back to the stream
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: Config, layer: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config, layer)
        self.ff_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ff = SwiGLU(config)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, start: int, cache: KVCache | None
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, start, cache)
        return x + self.ff(self.ff_norm(x))


class Decoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: the output layer scores tokens with the same vectors that embed
        # them. Saves V x C parameters and helps small models.
        self.head.weight = self.embed.weight
        cos, sin = rope_tables(config.head_dim, config.max_seq_len, config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # Residual projections start smaller, scaled by depth, so that the variance of
        # the residual stream stays O(1) after summing 2 * n_layers contributions.
        for block in self.blocks:
            assert isinstance(block, Block)
            for proj in (block.attn.wo, block.ff.w2):
                nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: Tensor, start: int = 0, cache: KVCache | None = None) -> Tensor:
        """Logits (B, T, V) for tokens (B, T) occupying positions start..start+T."""
        if start + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, start, cache)
        return self.head(self.norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

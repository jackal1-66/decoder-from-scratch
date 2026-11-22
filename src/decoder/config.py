"""Model configuration, shared by the PyTorch (``decoder``) and MLX (``decoder_mlx``) models.

It has no framework imports, so either implementation can use it without the other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256  # byte-level: every byte is a token
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2  # 3 query heads share each key/value head
    d_ff: int = 1024
    max_seq_len: int = 256
    rope_base: float = 10_000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE rotates pairs of dimensions: head_dim must be even")

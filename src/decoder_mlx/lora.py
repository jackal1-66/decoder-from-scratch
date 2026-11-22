"""LoRA for MLX: ``LoRALinear`` (merge, unmerge) and ``apply_lora``, as in ``decoder.lora``.

See ``decoder.lora`` for the method. Parameter names and shapes are the same:
``lora_a`` (r, d_in), ``lora_b`` (d_out, r), base weight at ``base.weight``.

Freezing differs from PyTorch: MLX arrays carry no ``requires_grad``. A module's
``freeze()`` marks its parameters as excluded from ``trainable_parameters()``, and
``nn.value_and_grad`` differentiates only the trainable ones.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update (α / r) B A."""

    def __init__(
        self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        d_out, d_in = base.weight.shape
        self.base = base
        self.base.freeze()
        self.scaling = alpha / rank
        # PyTorch's kaiming_uniform_(a=sqrt(5)), nn.Linear's default: U(±1/sqrt(d_in)).
        bound = 1 / math.sqrt(d_in)
        self.lora_a = mx.random.uniform(-bound, bound, (rank, d_in))
        self.lora_b = mx.zeros((d_out, rank))  # zero: the update starts at exactly zero
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def __call__(self, x: mx.array) -> mx.array:
        out: mx.array = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return out

    def delta_weight(self) -> mx.array:
        return (self.lora_b @ self.lora_a) * self.scaling

    def merge(self) -> None:
        """Fold the update into the base weight: inference then costs what the base costs."""
        if not self.merged:
            self.base.weight = self.base.weight + self.delta_weight()
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            self.base.weight = self.base.weight - self.delta_weight()
            self.merged = False


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] = ("wq", "wk", "wv", "wo"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every ``nn.Linear`` whose attribute name is in ``targets`` by a LoRALinear.

    Every other parameter is frozen. Returns the dotted names of the replaced layers.
    """
    model.freeze()
    replaced = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.items()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return replaced

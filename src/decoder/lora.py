"""LoRA: low-rank adaptation of a frozen linear layer.

A fine-tuning update ΔW to a d_out x d_in matrix is replaced by a rank-r factorisation
ΔW = (α / r) · B A, with A (r x d_in) and B (d_out x r):

* Only A and B train: r (d_in + d_out) parameters instead of d_in d_out. With r = 8 and
  d = 384 that is 4 % of the layer; at d = 4096, 0.4 %.
* B starts at zero, so the adapted model starts *exactly* at the pretrained one, and
  early training cannot destroy it. A starts random (Kaiming), otherwise the gradient
  of B (which is proportional to A x) would be zero too and nothing would move.
* α / r decouples the learning rate from the rank: doubling r with α fixed halves each
  direction's contribution, so a learning rate tuned at one rank transfers to another.
* After training, ``merge`` adds (α / r) B A into the base weight: inference then costs
  exactly what the base model costs. ``unmerge`` subtracts it again, so one base model
  can serve several adapters.

Why it works at all: fine-tuning updates of pretrained models are empirically close to
low-rank (Hu et al., 2021); the pretrained weights already contain most of what is
needed, and adaptation moves them in a few directions.
"""

import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # nn.Linear's own default
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return out

    def delta_weight(self) -> Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] = ("wq", "wk", "wv", "wo"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every nn.Linear whose attribute name is in ``targets`` by a LoRALinear.

    Every other parameter is frozen. Returns the dotted names of the replaced layers.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return replaced

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from decoder.model import Block, Config, Decoder, KVCache, RMSNorm, apply_rope, rope_tables


def test_output_shape(model: Decoder, config: Config) -> None:
    tokens = torch.randint(0, config.vocab_size, (3, 10))
    assert model(tokens).shape == (3, 10, config.vocab_size)


def test_causality_future_tokens_do_not_change_past_logits(model: Decoder, config: Config) -> None:
    a = torch.randint(0, config.vocab_size, (1, 12))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 1) % config.vocab_size  # change only positions 7..11
    la, lb = model(a), model(b)
    torch.testing.assert_close(la[:, :7], lb[:, :7])
    assert not torch.allclose(la[:, 7:], lb[:, 7:])


def test_incremental_decoding_with_cache_matches_full_forward(
    model: Decoder, config: Config
) -> None:
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    full = model(tokens)
    cache = KVCache(config, batch=2, device=tokens.device, dtype=torch.float32)
    prefill = model(tokens[:, :5], start=0, cache=cache)
    steps = [model(tokens[:, t : t + 1], start=t, cache=cache) for t in range(5, 16)]
    torch.testing.assert_close(torch.cat([prefill, *steps], dim=1), full, atol=1e-5, rtol=1e-5)


def test_attention_matches_pytorch_reference(config: Config) -> None:
    # With K = H (plain multi-head attention) our masked softmax must equal PyTorch's
    # fused scaled_dot_product_attention with is_causal=True.
    torch.manual_seed(1)
    mha = Config(**{**config.__dict__, "n_kv_heads": config.n_heads})
    block = Decoder(mha).blocks[0]
    assert isinstance(block, Block)
    attn = block.attn
    x = torch.randn(2, 9, mha.d_model)
    cos, sin = rope_tables(mha.head_dim, mha.max_seq_len, mha.rope_base)
    ours = attn(x, cos, sin, start=0, cache=None)
    B, T, H, D = 2, 9, mha.n_heads, mha.head_dim
    q = apply_rope(attn.wq(x).view(B, T, H, D).transpose(1, 2), cos[:T], sin[:T])
    k = apply_rope(attn.wk(x).view(B, T, H, D).transpose(1, 2), cos[:T], sin[:T])
    v = attn.wv(x).view(B, T, H, D).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(ours, attn.wo(ref.transpose(1, 2).reshape(B, T, H * D)))


def test_rope_scores_depend_only_on_relative_position() -> None:
    D = 16
    cos, sin = rope_tables(D, 64, 10_000.0)
    q, k = torch.randn(D), torch.randn(D)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q.view(1, 1, 1, D), cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k.view(1, 1, 1, D), cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    assert score(10, 7) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(10, 7) == pytest.approx(score(3, 0), abs=1e-4)
    assert score(10, 7) != pytest.approx(score(10, 2), abs=1e-4)


def test_rope_is_a_rotation_and_preserves_norms() -> None:
    cos, sin = rope_tables(8, 16, 10_000.0)
    x = torch.randn(1, 1, 16, 8)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rmsnorm_gives_unit_rms() -> None:
    x = 5 * torch.randn(4, 7, 32)
    y = RMSNorm(32, eps=1e-6)(x)
    torch.testing.assert_close(y.pow(2).mean(-1).sqrt(), torch.ones(4, 7), atol=1e-4, rtol=0)


def test_initial_loss_is_close_to_uniform(model: Decoder, config: Config) -> None:
    # Small initial weights -> near-uniform predictions -> next-token loss ~ ln(V) on
    # random tokens. Much larger would mean a bad initialisation; much smaller, a leak.
    # (Scoring position t against token t instead of t+1 gives ~3.3 < ln 50 = 3.9 here:
    # with tied embeddings the residual stream still carries the input token.)
    tokens = torch.randint(0, config.vocab_size, (8, 32))
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
    assert float(loss) == pytest.approx(math.log(config.vocab_size), rel=0.05)


def test_can_overfit_a_single_batch(config: Config) -> None:
    torch.manual_seed(0)
    model = Decoder(config)
    tokens = torch.randint(0, config.vocab_size, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.1


@pytest.mark.parametrize(
    "fields",
    [
        {"d_model": 30, "n_heads": 4},
        {"n_heads": 6, "n_kv_heads": 4},
        {"n_heads": 0},
        {"n_layers": -1},
        {"dropout": 1.0},
        {"norm_eps": 0.0},
    ],
)
def test_config_validation(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Config(**fields)

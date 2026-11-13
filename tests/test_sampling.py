import torch

from decoder.model import Config, Decoder
from decoder.sampling import filter_logits, generate, sample_next


def test_greedy_is_argmax() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 2.9]])
    assert sample_next(logits, temperature=0.0).tolist() == [1, 0]
    assert sample_next(logits, top_k=1).tolist() == [1, 0]


def test_top_k_keeps_k_largest() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    kept = torch.isfinite(filter_logits(logits, top_k=2))
    assert kept.tolist() == [[False, True, True, False]]


def test_top_p_keeps_smallest_nucleus() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()

    def kept(p: float) -> list[list[bool]]:
        return torch.isfinite(filter_logits(logits, top_p=p)).tolist()

    assert kept(0.5) == [[True, False, False, False]]
    assert kept(0.7) == [[True, True, False, False]]
    assert kept(0.9) == [[True, True, True, False]]
    assert kept(1e-9) == [[True, False, False, False]]  # the top token always survives


def test_sampling_follows_the_distribution() -> None:
    probs = torch.tensor([0.6, 0.3, 0.1])
    gen = torch.Generator().manual_seed(0)
    draws = sample_next(probs.log().expand(20_000, 3), generator=gen)
    freq = torch.bincount(draws, minlength=3).float() / 20_000
    torch.testing.assert_close(freq, probs, atol=0.015, rtol=0)


def test_low_temperature_concentrates_on_the_mode() -> None:
    logits = torch.tensor([1.0, 0.5, 0.0]).expand(5_000, 3)
    gen = torch.Generator().manual_seed(0)
    cold = sample_next(logits, temperature=0.1, generator=gen)
    hot = sample_next(logits, temperature=10.0, generator=gen)
    assert (cold == 0).float().mean() > 0.99
    assert (hot == 0).float().mean() < 0.4


def test_cached_and_uncached_generation_agree(model: Decoder, config: Config) -> None:
    prompt = torch.randint(0, config.vocab_size, (2, 5))
    with_cache = generate(model, prompt, max_new_tokens=20, temperature=0.0)
    without = generate(model, prompt, max_new_tokens=20, temperature=0.0, use_cache=False)
    assert torch.equal(with_cache, without)
    assert torch.equal(with_cache[:, :5], prompt)

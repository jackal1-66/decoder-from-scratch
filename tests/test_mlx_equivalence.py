"""The PyTorch and MLX models compute the same function (float32, CPU).

Tolerance: max |logit difference| <= TOLERANCE x max |logit|. Float32 rounding error grows
with the size of the values, so the bound is relative to the logit scale.

Measured with scripts/mlx_equivalence.py on the default 9.5M-parameter config, random and
trained weights: the frameworks differ by at most 1.3e-6 of the largest logit, the same
as either framework's error against a float64 reference (1.0e-6 PyTorch, 1.4e-6 MLX),
about 10 float32 ulps (epsilon 1.2e-7). On the small config of these tests the largest
difference is 2.1e-6 (trained weights, fast kernels). TOLERANCE leaves a margin of about
5x over that, and a real bug, the interleaved RoPE layout, exceeds it 100,000-fold
(test_tolerance_detects_the_wrong_rope_layout).
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from decoder.config import Config
from decoder.lora import LoRALinear, apply_lora
from decoder.model import Decoder
from decoder.sampling import generate
from decoder_mlx.convert import torch_to_mlx
from decoder_mlx.lora import LoRALinear as MLXLoRALinear
from decoder_mlx.lora import apply_lora as mlx_apply_lora
from decoder_mlx.model import Decoder as MLXDecoder
from decoder_mlx.model import KVCache
from decoder_mlx.sampling import generate as mlx_generate

TOLERANCE = 1e-5


def relative_difference(mlx_out: mx.array, torch_out: torch.Tensor) -> float:
    reference = torch_out.detach().double().numpy()
    return float(np.abs(np.array(mlx_out) - reference).max() / np.abs(reference).max())


def to_mlx(model: Decoder, use_fast_kernels: bool = True) -> MLXDecoder:
    mlx_model = MLXDecoder(model.config, use_fast_kernels)
    mlx_model.load_weights(list(torch_to_mlx(model.state_dict(), model.config).items()))
    mlx_model.eval()
    return mlx_model


def trained(config: Config) -> Decoder:
    """A model trained briefly on a repetitive text, so that its logits are sharp."""
    torch.manual_seed(0)
    model = Decoder(config)
    data = (
        torch.tensor(list(b"the cat sat on the mat. " * 20), dtype=torch.long) % config.vocab_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(100):
        starts = torch.randint(0, len(data) - 33, (8,))
        batch = torch.stack([data[s : s + 33] for s in starts])
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def random_weights(config: Config) -> Decoder:
    torch.manual_seed(0)
    return Decoder(config).eval()


@pytest.fixture(params=[random_weights, trained], ids=["random", "trained"])
def torch_model(request: pytest.FixtureRequest, config: Config) -> Decoder:
    make: Callable[[Config], Decoder] = request.param
    return make(config)


def tokens(config: Config, batch: int = 4, length: int = 32) -> torch.Tensor:
    return torch.randint(
        0, config.vocab_size, (batch, length), generator=torch.Generator().manual_seed(1)
    )


@pytest.mark.parametrize("use_fast_kernels", [True, False], ids=["fast", "explicit"])
def test_same_weights_give_the_same_logits(torch_model: Decoder, use_fast_kernels: bool) -> None:
    x = tokens(torch_model.config)
    with torch.no_grad():
        expected = torch_model(x)
    actual = to_mlx(torch_model, use_fast_kernels)(mx.array(x.numpy()))
    assert relative_difference(actual, expected) <= TOLERANCE


def test_trained_model_has_sharp_logits(config: Config) -> None:
    # Guards the premise of the "trained" cases: logits much larger than at initialisation
    # (measured: 6.6 vs 0.8), where float32 errors are correspondingly larger.
    sharp, initial = trained(config), random_weights(config)
    with torch.no_grad():
        assert sharp(tokens(config)).abs().max() > 5 * initial(tokens(config)).abs().max()


def test_cached_mlx_decoding_matches_the_full_pytorch_forward(torch_model: Decoder) -> None:
    config = torch_model.config
    x = tokens(config, batch=2, length=16)
    with torch.no_grad():
        expected = torch_model(x)
    mlx_model, xm = to_mlx(torch_model), mx.array(x.numpy())
    cache = KVCache(config, batch=2)
    outputs = [mlx_model(xm[:, :5], start=0, cache=cache)]
    outputs += [mlx_model(xm[:, t : t + 1], start=t, cache=cache) for t in range(5, 16)]
    assert relative_difference(mx.concatenate(outputs, axis=1), expected) <= TOLERANCE


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no-cache"])
def test_greedy_generation_gives_identical_tokens(torch_model: Decoder, use_cache: bool) -> None:
    prompt = tokens(torch_model.config, batch=3, length=4)
    expected = generate(torch_model, prompt, max_new_tokens=24, temperature=0.0)
    actual = mlx_generate(
        to_mlx(torch_model), mx.array(prompt.numpy()), 24, 0.0, use_cache=use_cache
    )
    assert np.array(actual).tolist() == expected.tolist()


def test_lora_agrees_across_frameworks_unmerged_and_merged(config: Config) -> None:
    model = random_weights(config)
    apply_lora(model, rank=4)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_b, std=0.1)  # pretend we trained
    mlx_model = MLXDecoder(config)
    mlx_apply_lora(mlx_model, rank=4)
    # Parameter names agree, LoRA ones included, so the PyTorch state loads as it is.
    state = {n: mx.array(t.numpy()) for n, t in model.state_dict().items() if n != "head.weight"}
    mlx_model.load_weights(list(state.items()), strict=True)
    mlx_model.eval()
    x = tokens(config)
    with torch.no_grad():
        expected = model(x)
    assert relative_difference(mlx_model(mx.array(x.numpy())), expected) <= TOLERANCE
    for _, module in mlx_model.named_modules():
        if isinstance(module, MLXLoRALinear):
            module.merge()
    assert relative_difference(mlx_model(mx.array(x.numpy())), expected) <= TOLERANCE


def test_lora_is_the_identity_at_initialisation_in_both(config: Config) -> None:
    model = random_weights(config)
    x = tokens(config)
    with torch.no_grad():
        before = model(x)
    mlx_model = to_mlx(model)
    mlx_apply_lora(mlx_model, rank=4)
    apply_lora(model, rank=4)
    with torch.no_grad():
        assert torch.equal(model(x), before)
    assert relative_difference(mlx_model(mx.array(x.numpy())), before) <= TOLERANCE


def test_tolerance_detects_the_wrong_rope_layout(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = trained(config)
    x = tokens(config)
    with torch.no_grad():
        expected = model(x)
    rope = mx.fast.rope

    def interleaved_rope(*args: Any, **kwargs: Any) -> mx.array:
        return rope(*args, **{**kwargs, "traditional": True})

    monkeypatch.setattr(mx.fast, "rope", interleaved_rope)
    assert relative_difference(to_mlx(model)(mx.array(x.numpy())), expected) > 100 * TOLERANCE

"""The properties checked in test_lora.py, for MLX."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from decoder.config import Config
from decoder_mlx.lora import LoRALinear, apply_lora
from decoder_mlx.model import Decoder, flat_parameters


def assert_close(actual: mx.array, expected: mx.array, atol: float) -> None:
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0, atol=atol)


def trained_lora_layer() -> LoRALinear:
    mx.random.seed(0)
    layer = LoRALinear(nn.Linear(16, 8), rank=4)
    layer.lora_b = mx.random.normal(layer.lora_b.shape)  # pretend we trained
    return layer


def test_fresh_lora_layer_is_the_identity_update() -> None:
    mx.random.seed(0)
    base = nn.Linear(16, 8)
    x = mx.random.normal((5, 16))
    assert mx.array_equal(LoRALinear(base, rank=4)(x), base(x)).item()


def test_only_lora_parameters_train_and_count_is_r_times_in_plus_out(config: Config) -> None:
    model = Decoder(config)
    replaced = apply_lora(model, rank=4)
    assert len(replaced) == 4 * config.n_layers
    trainable = sum(p.size for p in flat_parameters(model.trainable_parameters()).values())
    d, kv = config.d_model, config.n_kv_heads * config.head_dim
    per_layer = 4 * ((d + d) + (d + kv) + (d + kv) + (d + d))
    assert trainable == per_layer * config.n_layers


def test_merge_and_unmerge_are_exact_inverses() -> None:
    layer = trained_lora_layer()
    x = mx.random.normal((5, 16))
    before = layer(x)
    original = layer.base.weight
    layer.merge()
    assert_close(layer(x), before, atol=1e-5)
    layer.unmerge()
    assert_close(layer.base.weight, original, atol=1e-6)


def test_merging_a_model_does_not_change_its_output(config: Config) -> None:
    mx.random.seed(0)
    model = Decoder(config)
    model.eval()
    apply_lora(model, rank=4)
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_b = 0.1 * mx.random.normal(module.lora_b.shape)
    tokens = mx.random.randint(0, config.vocab_size, (2, 16))
    before = model(tokens)
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()
    assert_close(model(tokens), before, atol=1e-5)


def test_training_moves_only_the_adapters(config: Config) -> None:
    mx.random.seed(0)
    model = Decoder(config)
    apply_lora(model, rank=4)
    before = flat_parameters(model.parameters())
    tokens = mx.random.randint(0, config.vocab_size, (2, 16))
    optimizer = optim.AdamW(learning_rate=1e-2)

    def loss_fn(model: Decoder, tokens: mx.array) -> mx.array:
        return mx.mean(mx.logsumexp(model(tokens), axis=-1))

    _, grads = nn.value_and_grad(model, loss_fn)(model, tokens)
    optimizer.update(model, grads)
    after = flat_parameters(model.parameters())
    moved = {name for name in before if not mx.array_equal(before[name], after[name]).item()}
    assert moved
    assert all(".lora_" in name for name in moved)

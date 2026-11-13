import torch
from torch import nn

from decoder.lora import LoRALinear, apply_lora
from decoder.model import Config, Decoder


def test_fresh_lora_layer_is_the_identity_update() -> None:
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    layer = LoRALinear(base, rank=4)
    x = torch.randn(5, 16)
    torch.testing.assert_close(layer(x), base(x))


def test_only_lora_parameters_train_and_count_is_r_times_in_plus_out(config: Config) -> None:
    model = Decoder(config)
    replaced = apply_lora(model, rank=4)
    assert len(replaced) == 4 * config.n_layers
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d, kv = config.d_model, config.n_kv_heads * config.head_dim
    per_layer = 4 * ((d + d) + (d + kv) + (d + kv) + (d + d))
    assert trainable == per_layer * config.n_layers


def test_merge_and_unmerge_are_exact_inverses() -> None:
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 8), rank=4)
    nn.init.normal_(layer.lora_b)  # pretend we trained
    x = torch.randn(5, 16)
    before = layer(x)
    original = layer.base.weight.clone()
    layer.merge()
    torch.testing.assert_close(layer(x), before, atol=1e-5, rtol=1e-5)
    layer.unmerge()
    torch.testing.assert_close(layer.base.weight, original, atol=1e-6, rtol=0)


def test_training_moves_only_the_adapters(config: Config) -> None:
    torch.manual_seed(0)
    model = Decoder(config)
    apply_lora(model, rank=4)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    tokens = torch.randint(0, config.vocab_size, (2, 16))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = model(tokens).logsumexp(-1).mean()
    loss.backward()
    opt.step()
    for n, p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n])

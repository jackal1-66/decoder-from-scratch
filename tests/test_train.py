import json
from pathlib import Path

import pytest
import torch

from decoder.lora import apply_lora
from decoder.model import Config, Decoder
from decoder.train import TrainConfig, eval_batches, load_bytes, lr_at, split, train


def test_gutenberg_header_and_footer_are_stripped(tmp_path: Path) -> None:
    raw = (
        "licence text\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nThe book.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nmore licence\n"
    )
    path = tmp_path / "book.txt"
    path.write_text(raw)
    assert bytes(load_bytes(path, gutenberg=True).tolist()) == b"The book.\n"


def test_split_is_contiguous() -> None:
    data = torch.arange(100)
    train, held = split(data, 0.2)
    assert train.tolist() == list(range(80)) and held.tolist() == list(range(80, 100))


def test_schedule_warms_up_then_decays_to_floor() -> None:
    cfg = TrainConfig(steps=1000, warmup_steps=100, learning_rate=1.0, min_lr_fraction=0.1)
    assert lr_at(0, cfg) == 0.01
    assert lr_at(99, cfg) == 1.0
    assert abs(lr_at(1000, cfg) - 0.1) < 1e-9


def test_training_learns_a_repetitive_text_on_cpu(tmp_path: Path) -> None:
    data = torch.tensor(list(b"abcdefgh" * 400), dtype=torch.long)
    train_data, val = split(data, 0.1)
    config = Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=32)
    model = Decoder(config)
    cfg = TrainConfig(steps=150, batch_size=8, learning_rate=3e-3, warmup_steps=10, eval_every=50)
    final = train(model, train_data, {"val": val}, cfg, tmp_path, device="cpu")
    assert final["val"] < 0.5  # 8 equiprobable bytes would be 3 bits; the pattern is ~0
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if "val_bpb" in r] == [0, 50, 100, 150]
    assert [r["step"] for r in records if "train_bpb" in r] == [0, 50, 100]


@pytest.mark.parametrize("grad_accum", [0, 3])
def test_grad_accum_must_divide_the_batch(grad_accum: int) -> None:
    with pytest.raises(ValueError, match="grad_accum"):
        TrainConfig(batch_size=8, grad_accum=grad_accum)


def test_gradient_accumulation_gives_the_full_batch_update(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    train_data, val = split(torch.randint(0, 256, (4_000,), generator=generator), 0.1)
    config = Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=32)
    torch.manual_seed(0)
    initial = Decoder(config).state_dict()
    curves, finals = [], []
    for grad_accum in (1, 4):
        model = Decoder(config)
        model.load_state_dict(initial)
        cfg = TrainConfig(steps=20, batch_size=8, warmup_steps=5, eval_every=10, log_every=1,
                          grad_accum=grad_accum)  # fmt: skip
        out = tmp_path / str(grad_accum)
        finals.append(train(model, train_data, {"val": val}, cfg, out, device="cpu")["val"])
        lines = (out / "log.jsonl").read_text().splitlines()
        curves.append([r["train_bpb"] for r in map(json.loads, lines) if "train_bpb" in r])
    assert len(curves[0]) == 20
    assert curves[1] == pytest.approx(curves[0], rel=1e-5)
    assert finals[1] == pytest.approx(finals[0], rel=1e-5)


def test_lora_finetune_leaves_base_weights_untouched(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = Config(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=32)
    model = Decoder(config)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    apply_lora(model, targets=("wq", "wv"), rank=2)
    data = torch.tensor(list(b"xyz" * 300), dtype=torch.long)
    train(model, data, {}, TrainConfig(steps=20, batch_size=4, warmup_steps=2, eval_every=100),
          tmp_path, device="cpu")  # fmt: skip
    state = model.state_dict()
    for name, value in base.items():
        lora_name = name.replace(".wq.", ".wq.base.").replace(".wv.", ".wv.base.")
        torch.testing.assert_close(state[lora_name], value)


def test_eval_batches_are_consecutive_non_overlapping_windows() -> None:
    data = torch.arange(100)
    batches = list(eval_batches(data, seq_len=10, max_windows=5, batch_size=2))
    assert [x.shape[0] for x, _ in batches] == [2, 2, 1]
    x = torch.cat([x for x, _ in batches])
    y = torch.cat([y for _, y in batches])
    assert x[:, 0].tolist() == [0, 10, 20, 30, 40]
    assert torch.equal(y, x + 1)

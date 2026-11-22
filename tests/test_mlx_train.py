"""The MLX training loop: same curve as decoder.train, exact resume, and the ported tests."""

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest
import torch

import decoder_mlx.train as mlx_train_module
from decoder.config import Config
from decoder.model import Decoder as TorchDecoder
from decoder.train import TrainConfig, split
from decoder.train import evaluate as torch_evaluate
from decoder.train import lr_at as decoder_lr_at
from decoder.train import train as torch_train
from decoder_mlx.checkpoint import CheckpointError, save_model
from decoder_mlx.convert import torch_to_mlx
from decoder_mlx.lora import apply_lora
from decoder_mlx.model import Decoder, flat_parameters
from decoder_mlx.train import evaluate, train

# Relative, as for the logits in test_mlx_equivalence.py; measured: <= 5.3e-7 over 50 steps.
TOLERANCE = 1e-5

SMALL = Config(vocab_size=256, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64,
               max_seq_len=32)  # fmt: skip


def text_bytes() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    words = [b"the ", b"cat ", b"sat ", b"on ", b"a ", b"mat. "]
    order = torch.randint(0, len(words), (2_000,), generator=generator).tolist()
    return torch.tensor(list(b"".join(words[i] for i in order)), dtype=torch.long)


def converted(model: TorchDecoder) -> Decoder:
    mlx_model = Decoder(model.config)
    mlx_model.load_weights(list(torch_to_mlx(model.state_dict(), model.config).items()))
    return mlx_model


def records(out: Path, key: str) -> dict[int, float]:
    lines = (out / "log.jsonl").read_text().splitlines()
    return {r["step"]: r[key] for r in map(json.loads, lines) if key in r}


def test_loss_curve_matches_pytorch_from_the_same_weights_and_batches(tmp_path: Path) -> None:
    torch.manual_seed(0)
    torch_model = TorchDecoder(SMALL)
    mlx_model = converted(torch_model)
    train_data, val = split(text_bytes(), 0.1)
    cfg = TrainConfig(steps=20, batch_size=8, warmup_steps=5, eval_every=10, log_every=1)
    torch_final = torch_train(torch_model, train_data, {"val": val}, cfg, tmp_path / "t", "cpu")
    mlx_final = train(mlx_model, train_data, {"val": val}, cfg, tmp_path / "m")
    torch_curve, mlx_curve = (
        records(tmp_path / "t", "train_bpb"),
        records(tmp_path / "m", "train_bpb"),
    )
    assert torch_curve.keys() == mlx_curve.keys() == set(range(20))
    for step, expected in torch_curve.items():
        assert mlx_curve[step] == pytest.approx(expected, rel=TOLERANCE), step
    assert mlx_final["val"] == pytest.approx(torch_final["val"], rel=TOLERANCE)


def test_evaluation_matches_pytorch(model: TorchDecoder) -> None:
    data = torch.randint(
        0, model.config.vocab_size, (2_000,), generator=torch.Generator().manual_seed(0)
    )
    expected = torch_evaluate(model, data, model.config.max_seq_len, "cpu")
    assert evaluate(converted(model), data, model.config.max_seq_len) == pytest.approx(
        expected, rel=TOLERANCE
    )


def run(
    out: Path,
    resume: bool = False,
    crash_at: int | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[Decoder, dict[str, float]]:
    config = Config(**{**SMALL.__dict__, "dropout": 0.2})  # dropout: tests the reseeding
    mx.random.seed(0)
    model = Decoder(config)
    if crash_at is not None and monkeypatch is not None:

        def crashing_lr_at(step: int, cfg: TrainConfig) -> float:
            if step == crash_at:
                raise KeyboardInterrupt
            return decoder_lr_at(step, cfg)

        monkeypatch.setattr(mlx_train_module, "lr_at", crashing_lr_at)
    train_data, val = split(text_bytes(), 0.1)
    cfg = TrainConfig(steps=12, batch_size=4, warmup_steps=2, eval_every=5, log_every=1)
    return model, train(model, train_data, {"val": val}, cfg, out, resume=resume)


def test_resume_after_a_crash_gives_the_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight, final = run(tmp_path / "straight")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resumed", crash_at=8, monkeypatch=monkeypatch)  # last checkpoint: 5
    monkeypatch.undo()
    resumed, resumed_final = run(tmp_path / "resumed", resume=True)

    expected, actual = flat_parameters(straight.parameters()), flat_parameters(resumed.parameters())
    assert all(mx.array_equal(expected[n], actual[n]).item() for n in expected)
    assert resumed_final == final

    def log(out: Path) -> list[dict[str, Any]]:
        lines = (out / "log.jsonl").read_text().splitlines()
        return [{k: v for k, v in json.loads(line).items() if k != "seconds"} for line in lines]

    assert log(tmp_path / "resumed") == log(tmp_path / "straight")


def test_resume_refuses_different_settings(tmp_path: Path) -> None:
    mx.random.seed(0)
    model = Decoder(SMALL)
    data = text_bytes()
    train(model, data, {}, TrainConfig(steps=2, batch_size=2, warmup_steps=1), tmp_path)
    changed = TrainConfig(steps=4, batch_size=2, warmup_steps=1)
    with pytest.raises(CheckpointError, match="resume with the same settings"):
        train(Decoder(SMALL), data, {}, changed, tmp_path, resume=True)
    with pytest.raises(CheckpointError, match="does not exist"):
        train(Decoder(SMALL), data, {}, changed, tmp_path / "empty", resume=True)


def test_training_learns_a_repetitive_text(tmp_path: Path) -> None:
    data = torch.tensor(list(b"abcdefgh" * 400), dtype=torch.long)
    train_data, val = split(data, 0.1)
    mx.random.seed(0)
    model = Decoder(SMALL)
    cfg = TrainConfig(steps=150, batch_size=8, learning_rate=3e-3, warmup_steps=10, eval_every=50)
    final = train(model, train_data, {"val": val}, cfg, tmp_path)
    assert final["val"] < 0.5  # 8 equiprobable bytes would be 3 bits; the pattern is ~0
    assert sorted(records(tmp_path, "val_bpb")) == [0, 50, 100, 150]


def test_lora_finetune_leaves_base_weights_untouched(tmp_path: Path) -> None:
    mx.random.seed(0)
    model = Decoder(SMALL)
    base = flat_parameters(model.parameters())
    apply_lora(model, targets=("wq", "wv"), rank=2)
    data = torch.tensor(list(b"xyz" * 300), dtype=torch.long)
    train(model, data, {}, TrainConfig(steps=20, batch_size=4, warmup_steps=2, eval_every=100),
          tmp_path)  # fmt: skip
    after = flat_parameters(model.parameters())
    for name, value in base.items():
        lora_name = name.replace(".wq.", ".wq.base.").replace(".wv.", ".wv.base.")
        assert mx.array_equal(after[lora_name], value).item(), name
    assert any(mx.any(a != 0).item() for n, a in after.items() if ".lora_b" in n)


def test_finetune_command_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mx.random.seed(0)
    save_model(Decoder(SMALL), tmp_path / "init.safetensors")
    text = tmp_path / "text.txt"
    text.write_bytes(b"the cat sat on the mat. " * 200)
    argv = ["finetune", "--init", str(tmp_path / "init.safetensors"), "--text", str(text),
            "--out", str(tmp_path / "run"), "--steps", "3", "--lora-rank", "2"]  # fmt: skip
    monkeypatch.setattr("sys.argv", ["decoder_mlx.train", *argv])
    mlx_train_module.main()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["framework"] == "mlx"
    assert summary["final_bpb"].keys() == {"dev", "test"}
    assert 0 < summary["trainable_parameters"] < summary["parameters"]

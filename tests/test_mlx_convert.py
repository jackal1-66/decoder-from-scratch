"""Checkpoint conversion: exact round trips, and actionable errors without partial output."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest
import torch

from decoder.config import Config
from decoder.lora import apply_lora
from decoder.model import Decoder
from decoder_mlx import checkpoint
from decoder_mlx.checkpoint import CheckpointError, load_model, load_weights, save_weights
from decoder_mlx.convert import (
    convert_mlx_to_torch,
    convert_torch_to_mlx,
    main,
    mlx_to_torch,
    torch_to_mlx,
)


def write_pt(path: Path, config: dict[str, Any], state: dict[str, torch.Tensor]) -> Path:
    """A checkpoint as decoder.train writes it."""
    torch.save({"config": config, "model": state}, path)
    return path


@pytest.fixture
def pt_file(tmp_path: Path, model: Decoder, config: Config) -> Path:
    return write_pt(tmp_path / "model.pt", dataclasses.asdict(config), model.state_dict())


def read_safetensors(path: Path) -> tuple[dict[str, mx.array], dict[str, str]]:
    loaded = mx.load(str(path), return_metadata=True)
    assert isinstance(loaded, tuple)
    return loaded


def assert_no_output(directory: Path, *expected: str) -> None:
    """Only the expected inputs are in ``directory``: no output, no temporary file."""
    assert sorted(p.name for p in directory.iterdir()) == sorted(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_mlx_torch_files_round_trip_exactly(
    tmp_path: Path, model: Decoder, config: Config, dtype: torch.dtype
) -> None:
    state = {name: t.to(dtype) for name, t in model.state_dict().items()}
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), state)
    convert_torch_to_mlx(src, tmp_path / "model.safetensors")
    convert_mlx_to_torch(tmp_path / "model.safetensors", tmp_path / "out.pt")

    back = torch.load(tmp_path / "out.pt", weights_only=True)
    assert back["config"] == dataclasses.asdict(config)
    assert back["model"].keys() == state.keys()
    for name, tensor in state.items():
        assert back["model"][name].dtype == dtype
        assert torch.equal(back["model"][name], tensor), name
    # The tied head is one tensor again, so training the reloaded model keeps it tied.
    reloaded = Decoder(config)
    reloaded.load_state_dict(back["model"])
    assert reloaded.head.weight is reloaded.embed.weight


def test_mlx_checkpoint_records_format_version_and_config(pt_file: Path, config: Config) -> None:
    dst = pt_file.with_name("model.safetensors")
    convert_torch_to_mlx(pt_file, dst)
    _, metadata = read_safetensors(dst)
    assert metadata["format"] == checkpoint.FORMAT_NAME
    assert metadata["format_version"] == str(checkpoint.FORMAT_VERSION)
    assert json.loads(metadata["config"]) == dataclasses.asdict(config)
    assert load_model(dst).config == config


def test_in_memory_conversion_round_trips_both_ways(model: Decoder, config: Config) -> None:
    weights = torch_to_mlx(model.state_dict(), config)
    assert "head.weight" not in weights
    again = torch_to_mlx(mlx_to_torch(weights, config), config)
    assert all(mx.array_equal(again[n], weights[n]).item() for n in weights)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("n_layers"), r"missing \['n_layers'\]"),
        (lambda c: c.update(n_experts=4), r"unknown \['n_experts'\]"),
        (lambda c: c.update(n_heads="4"), "n_heads must be int"),
        (lambda c: c.update(n_kv_heads=3), "n_heads must be divisible by n_kv_heads"),
    ],
)
def test_wrong_config_is_rejected(
    tmp_path: Path, model: Decoder, config: Config, change: Any, message: str
) -> None:
    raw = dataclasses.asdict(config)
    change(raw)
    src = write_pt(tmp_path / "in.pt", raw, model.state_dict())
    with pytest.raises(CheckpointError, match=message):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    assert_no_output(tmp_path, "in.pt")


def test_config_that_disagrees_with_the_weights_is_rejected(
    tmp_path: Path, model: Decoder, config: Config
) -> None:
    src = write_pt(tmp_path / "in.pt", {**dataclasses.asdict(config), "d_ff": 128},
                   model.state_dict())  # fmt: skip
    with pytest.raises(CheckpointError, match=r"ff.w1.weight: shape \(64, 32\).*\(128, 32\)"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    assert_no_output(tmp_path, "in.pt")


def test_missing_tensor_is_named(tmp_path: Path, model: Decoder, config: Config) -> None:
    state = model.state_dict()
    del state["blocks.1.attn.wv.weight"]
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), state)
    with pytest.raises(CheckpointError, match=r"missing tensors: blocks\.1\.attn\.wv\.weight"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    assert_no_output(tmp_path, "in.pt")


def test_shape_mismatch_is_named(tmp_path: Path, model: Decoder, config: Config) -> None:
    state = model.state_dict()
    state["norm.weight"] = torch.ones(config.d_model + 1)
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), state)
    with pytest.raises(CheckpointError, match=r"norm.weight: shape \(33,\), the config implies"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    assert_no_output(tmp_path, "in.pt")


def test_unsupported_dtype_is_rejected(tmp_path: Path, model: Decoder, config: Config) -> None:
    state = model.state_dict()
    state["norm.weight"] = state["norm.weight"].double()
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), state)
    with pytest.raises(CheckpointError, match=r"norm\.weight: dtype float64"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")


def test_untied_head_is_rejected(tmp_path: Path, model: Decoder, config: Config) -> None:
    state = model.state_dict()
    state["head.weight"] = state["head.weight"] + 1
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), state)
    with pytest.raises(CheckpointError, match="cannot represent an untied head"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    assert_no_output(tmp_path, "in.pt")


def test_unmerged_lora_adapters_are_rejected_with_a_hint(
    tmp_path: Path, model: Decoder, config: Config
) -> None:
    apply_lora(model, rank=2)
    src = write_pt(tmp_path / "in.pt", dataclasses.asdict(config), model.state_dict())
    with pytest.raises(CheckpointError, match="must be merged into the base weights"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")


@pytest.mark.parametrize("content", [b"", b"not a checkpoint", None])
def test_unreadable_input_is_reported(tmp_path: Path, content: bytes | None) -> None:
    src = tmp_path / "in.pt"
    if content is None:
        torch.save({"weights": torch.ones(2)}, src)  # readable, but not our format
    else:
        src.write_bytes(content)
    with pytest.raises(CheckpointError, match=r"in\.pt"):
        convert_torch_to_mlx(src, tmp_path / "out.safetensors")
    with pytest.raises(CheckpointError, match=r"in\.pt"):
        load_weights(src)


def test_newer_format_version_is_rejected(tmp_path: Path, model: Decoder, config: Config) -> None:
    path = tmp_path / "model.safetensors"
    save_weights(path, config, torch_to_mlx(model.state_dict(), config))
    arrays, metadata = read_safetensors(path)
    mx.save_safetensors(str(path), arrays, {**metadata, "format_version": "2"})
    with pytest.raises(CheckpointError, match=r"format version '2'.*upgrade"):
        load_weights(path)


def test_existing_output_is_kept_without_overwrite(pt_file: Path) -> None:
    dst = pt_file.with_name("out.safetensors")
    dst.write_bytes(b"keep me")
    with pytest.raises(CheckpointError, match="already exists"):
        convert_torch_to_mlx(pt_file, dst)
    assert dst.read_bytes() == b"keep me"
    convert_torch_to_mlx(pt_file, dst, overwrite=True)
    assert load_weights(dst)


def test_a_failed_write_leaves_no_partial_file(
    pt_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_half_then_fail(path: str, *args: Any) -> None:
        Path(path).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(mx, "save_safetensors", write_half_then_fail)
    with pytest.raises(OSError, match="disk full"):
        convert_torch_to_mlx(pt_file, pt_file.with_name("out.safetensors"))
    assert_no_output(pt_file.parent, "model.pt")


def test_cli_converts_and_reports_errors(pt_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dst = pt_file.with_name("model.safetensors")
    assert main(["torch-to-mlx", str(pt_file), str(dst)]) == 0
    assert main(["mlx-to-torch", str(dst), str(pt_file.with_name("back.pt"))]) == 0
    with pytest.raises(SystemExit) as exit_info:
        main(["torch-to-mlx", str(pt_file), str(dst)])
    assert exit_info.value.code == 1
    assert "decoder-convert: error:" in capsys.readouterr().err

"""decoder-benchmark: the report has every case, consistent statistics and a table."""

import dataclasses
import json
from pathlib import Path

from decoder.config import Config
from decoder_mlx.benchmark import main


def test_benchmark_writes_json_and_markdown(tmp_path: Path, config: Config) -> None:
    (tmp_path / "config.json").write_text(json.dumps(dataclasses.asdict(config)))
    argv = ["--out", str(tmp_path / "b.json"), "--markdown", str(tmp_path / "b.md"),
            "--config", str(tmp_path / "config.json"), "--torch-device", "cpu",
            "--batch", "2", "--new-tokens", "4", "--warmup", "1", "--repeats", "3"]  # fmt: skip
    assert main(argv) == 0
    report = json.loads((tmp_path / "b.json").read_text())
    cases = {(r["case"], r["framework"]) for r in report["results"]}
    assert cases == {
        ("forward", "pytorch"), ("forward", "mlx"), ("forward", "mlx compiled"),
        ("forward", "mlx explicit kernels"),
        ("generate, cache", "pytorch"), ("generate, cache", "mlx"),
        ("generate, no cache", "pytorch"), ("generate, no cache", "mlx"),
    }  # fmt: skip
    for r in report["results"]:
        assert len(r["seconds"]) == 3
        assert r["min_s"] <= r["median_s"] <= r["max_s"]
    assert {"torch", "mlx", "mlx_backend", "cpu"} <= report["environment"].keys()
    table = (tmp_path / "b.md").read_text()
    assert "on cpu; MLX" in table  # no device name on CPU, no stray space
    assert table.count("\n| ") == 9  # header + 8 rows

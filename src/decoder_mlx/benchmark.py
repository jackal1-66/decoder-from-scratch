"""Benchmark the PyTorch and the MLX decoder on this machine.

    decoder-benchmark --out benchmark.json --markdown benchmark.md

Same configuration (``--config``, default ``Config()``) and random weights, float32:

* forward: logits for a (batch, seq_len) batch, in PyTorch, in MLX, in MLX through
  ``mx.compile``, and in MLX without the ``mx.fast`` RoPE and attention kernels
  (``use_fast_kernels=False``);
* generate: greedy decoding of ``new_tokens`` tokens from a one-token prompt, with and
  without the KV cache, in PyTorch and in MLX.

Generation is not compiled: every decoding step has a new key length, and a new input
shape makes ``mx.compile`` trace and compile again.

Each case runs ``warmup`` times untimed, then ``repeats`` times timed; the device is
synchronised inside the timed region (``mx.eval``; ``torch.cuda/mps.synchronize``), since
both frameworks return before the work is done. Reported: median, interquartile range,
min and max, and tokens per second at the median. PyTorch runs on CUDA if available,
else MPS (Apple silicon), else CPU; MLX on its default device (the GPU on Apple silicon).
"""

import argparse
import dataclasses
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from decoder.config import Config
from decoder.model import Decoder as TorchDecoder
from decoder.sampling import generate as torch_generate
from decoder_mlx.checkpoint import config_from_dict
from decoder_mlx.convert import torch_to_mlx
from decoder_mlx.model import Decoder
from decoder_mlx.sampling import generate


@dataclasses.dataclass(frozen=True)
class Result:
    case: str
    framework: str
    device: str
    tokens: int
    seconds: list[float]

    def summary(self) -> dict[str, Any]:
        q1, median, q3 = statistics.quantiles(self.seconds, n=4, method="inclusive")
        return {
            "case": self.case, "framework": self.framework, "device": self.device,
            "tokens": self.tokens, "repeats": len(self.seconds), "seconds": self.seconds,
            "median_s": median, "iqr_s": q3 - q1, "min_s": min(self.seconds),
            "max_s": max(self.seconds), "tokens_per_s": self.tokens / median,
        }  # fmt: skip


def time_runs(run: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        run()
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return seconds


def torch_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def cpu_name() -> str:
    if platform.system() == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def mlx_backend() -> str:
    if mx.metal.is_available():
        return "metal"
    return "cuda" if mx.default_device() == mx.gpu else "cpu"


def environment(device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": version("torch"),
        "torch_device": str(device),
        "torch_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_threads": torch.get_num_threads(),
        "mlx": version("mlx"),
        "mlx_device": str(mx.default_device()),
        "mlx_backend": mlx_backend(),
        "numpy": np.__version__,
    }


def run_benchmarks(
    config: Config, batch: int, new_tokens: int, warmup: int, repeats: int, device: torch.device
) -> list[Result]:
    torch.manual_seed(0)
    torch_model = TorchDecoder(config).eval()
    weights = list(torch_to_mlx(torch_model.state_dict(), config).items())
    mlx_model = Decoder(config)
    mlx_model.load_weights(weights)
    mlx_model.eval()
    # RoPE and attention written out with ordinary operations instead of mx.fast.
    explicit_model = Decoder(config, use_fast_kernels=False)
    explicit_model.load_weights(weights)
    explicit_model.eval()
    torch_model.to(device)
    T = config.max_seq_len
    tokens = np.random.default_rng(0).integers(0, config.vocab_size, (batch, T))
    torch_tokens = torch.from_numpy(tokens).to(device)
    mlx_tokens = mx.array(tokens.astype(np.int32))
    # inputs=state: the compiled function reads the current weights instead of baking in
    # the ones it was traced with.
    compiled = mx.compile(lambda x: mlx_model(x), inputs=mlx_model.state)

    def torch_forward() -> None:
        with torch.no_grad():
            torch_model(torch_tokens)
        synchronize(device)

    mlx_dev, torch_dev = mlx_backend(), str(device)
    results = [
        Result("forward", "pytorch", torch_dev, batch * T,
               time_runs(torch_forward, warmup, repeats)),
        Result("forward", "mlx", mlx_dev, batch * T,
               time_runs(lambda: mx.eval(mlx_model(mlx_tokens)), warmup, repeats)),
        Result("forward", "mlx compiled", mlx_dev, batch * T,
               time_runs(lambda: mx.eval(compiled(mlx_tokens)), warmup, repeats)),
        Result("forward", "mlx explicit kernels", mlx_dev, batch * T,
               time_runs(lambda: mx.eval(explicit_model(mlx_tokens)), warmup, repeats)),
    ]  # fmt: skip

    torch_prompt = torch.full((batch, 1), ord("\n"), dtype=torch.long, device=device)
    mlx_prompt = mx.full((batch, 1), ord("\n"), dtype=mx.int32)
    for use_cache in (True, False):
        case = "generate, cache" if use_cache else "generate, no cache"

        def torch_run(use_cache: bool = use_cache) -> None:
            torch_generate(torch_model, torch_prompt, new_tokens, 0.0, use_cache=use_cache)
            synchronize(device)

        def mlx_run(use_cache: bool = use_cache) -> None:
            # generate evaluates every step and returns evaluated tokens.
            generate(mlx_model, mlx_prompt, new_tokens, 0.0, use_cache=use_cache)

        results.append(Result(case, "pytorch", torch_dev, batch * new_tokens,
                              time_runs(torch_run, warmup, repeats)))  # fmt: skip
        results.append(Result(case, "mlx", mlx_dev, batch * new_tokens,
                              time_runs(mlx_run, warmup, repeats)))  # fmt: skip
    return results


def markdown(report: dict[str, Any]) -> str:
    env, settings = report["environment"], report["settings"]
    torch_on = " ".join(filter(None, [env["torch_device"], env["torch_device_name"]]))
    lines = [
        f"Machine: {env['cpu']} ({env['machine']}); PyTorch {env['torch']} on {torch_on}; "
        f"MLX {env['mlx']} on {env['mlx_backend']}.",
        f"Model: {report['parameters'] / 1e6:.2f}M parameters, float32; batch "
        f"{settings['batch']}, forward length {report['config']['max_seq_len']}, "
        f"{settings['new_tokens']} new tokens; median of {settings['repeats']} runs after "
        f"{settings['warmup']} warm-up runs.",
        "",
        "| case | framework | device | median (ms) | IQR (ms) | min–max (ms) | tokens/s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['case']} | {r['framework']} | {r['device']} | {1e3 * r['median_s']:.1f} | "
            f"{1e3 * r['iqr_s']:.1f} | {1e3 * r['min_s']:.1f}–{1e3 * r['max_s']:.1f} | "
            f"{r['tokens_per_s']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="decoder-benchmark", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )  # fmt: skip
    p.add_argument("--out", type=Path, required=True, help="JSON report")
    p.add_argument("--markdown", type=Path, help="also write the table to this file")
    p.add_argument("--config", type=Path, help="JSON with every Config field (default Config())")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--torch-device", help="cpu, cuda, mps (default: the fastest available)")
    args = p.parse_args(argv)
    if args.repeats < 2 or args.warmup < 0 or args.batch < 1 or args.new_tokens < 1:
        p.error("need --repeats >= 2, --warmup >= 0, --batch >= 1, --new-tokens >= 1")
    config = config_from_dict(json.loads(args.config.read_text())) if args.config else Config()
    if args.new_tokens + 1 > config.max_seq_len:
        p.error(f"--new-tokens must be below max_seq_len ({config.max_seq_len})")

    device = torch_device(args.torch_device)
    results = run_benchmarks(config, args.batch, args.new_tokens, args.warmup, args.repeats, device)
    report = {
        "environment": environment(device),
        "config": dataclasses.asdict(config),
        "parameters": TorchDecoder(config).num_parameters(),
        "settings": {"batch": args.batch, "new_tokens": args.new_tokens,
                     "warmup": args.warmup, "repeats": args.repeats, "dtype": "float32"},
        "results": [r.summary() for r in results],
    }  # fmt: skip
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    table = markdown(report)
    if args.markdown:
        args.markdown.write_text(table)
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

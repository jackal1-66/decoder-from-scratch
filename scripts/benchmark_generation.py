"""Generation speed with and without the KV cache, and a text sample.

Two regimes:
* the trained 9.5M-parameter model, 256-token context: every decoding step is so cheap
  that its time is set by Python and kernel-launch overhead, not by arithmetic;
* a randomly initialised ~150M-parameter model with a 2048-token context (weights do not
  affect timing): recomputing the prefix becomes the cost, which the cache removes.

Usage: python scripts/benchmark_generation.py <model.pt>
Prints a Markdown table (tokens per second, median of 3 runs) and a sample.
"""

import statistics
import sys
import time

import torch

from decoder.model import Config, Decoder
from decoder.sampling import generate


def tokens_per_second(model: Decoder, batch: int, new: int, use_cache: bool) -> float:
    prompt = torch.full((batch, 1), ord("\n"), dtype=torch.long, device="cuda")
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate(model, prompt, new, temperature=0.0, use_cache=use_cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return batch * new / statistics.median(times)


def main() -> None:
    checkpoint = torch.load(sys.argv[1], map_location="cpu")
    model = Decoder(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model = model.cuda().half().eval()
    generate(model, torch.zeros((1, 1), dtype=torch.long, device="cuda"), 8)  # warm-up

    large_config = Config(d_model=1024, n_layers=12, n_heads=16, n_kv_heads=4, d_ff=2816,
                          max_seq_len=2048)  # fmt: skip
    large = Decoder(large_config).cuda().half().eval()

    print("| model | batch | new tokens | without cache (tok/s) | with cache (tok/s) | speed-up |")
    print("|---|---|---|---|---|---|")
    cases = [(model, "trained 9.5M", b, n) for b in (1, 16) for n in (64, 255)]
    cases += [
        (large, f"random {large.num_parameters() / 1e6:.0f}M", b, n)
        for b in (1, 8)
        for n in (256, 1024)
    ]
    for m, name, batch, new in cases:
        slow = tokens_per_second(m, batch, new, use_cache=False)
        fast = tokens_per_second(m, batch, new, use_cache=True)
        print(
            f"| {name} | {batch} | {new} | {slow:,.0f} | {fast:,.0f} | {fast / slow:.1f}x |",
            flush=True,
        )

    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.tensor([list(b"ROMEO:\n")], device="cuda")
    out = generate(model, prompt, 240, temperature=0.8, top_p=0.95, generator=gen)
    print("\nSample (temperature 0.8, top-p 0.95, seed 0):\n")
    print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()

import pytest
import torch

from decoder.model import Config, Decoder


@pytest.fixture
def config() -> Config:
    # Small enough for CPU tests; GQA on (4 query heads share 2 kv heads).
    return Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64,
                  max_seq_len=32)  # fmt: skip


@pytest.fixture
def model(config: Config) -> Decoder:
    torch.manual_seed(0)
    return Decoder(config).eval()

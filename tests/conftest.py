import pytest
import torch

from decoder.model import Config, Decoder

try:
    import mlx.core as mx

    from decoder_mlx.model import Decoder as MLXDecoder
except ImportError:
    # Without MLX (the CI job that checks the PyTorch package alone) skip collecting
    # the MLX tests instead of failing on their imports.
    collect_ignore_glob = ["test_mlx_*.py"]
else:

    @pytest.fixture
    def mlx_model(config: Config) -> MLXDecoder:
        mx.random.seed(0)
        model = MLXDecoder(config)
        model.eval()
        return model


@pytest.fixture
def config() -> Config:
    # Small enough for CPU tests; GQA on (4 query heads share 2 kv heads).
    return Config(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64,
                  max_seq_len=32)  # fmt: skip


@pytest.fixture
def model(config: Config) -> Decoder:
    torch.manual_seed(0)
    return Decoder(config).eval()

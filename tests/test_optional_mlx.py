"""MLX is optional: the PyTorch package works without it, and decoder_mlx says how to get it."""

import subprocess
import sys

# A None entry in sys.modules makes `import mlx...` raise ImportError, which simulates a
# missing MLX even where it is installed. Each check runs in a fresh interpreter because
# import state is process-wide.
HIDE_MLX = "import sys; sys.modules['mlx'] = None\n"


def run_without_mlx(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", HIDE_MLX + code], capture_output=True, text=True, check=False
    )


def test_torch_package_imports_without_mlx() -> None:
    result = run_without_mlx("import decoder.model, decoder.sampling, decoder.lora, decoder.train")
    assert result.returncode == 0, result.stderr


def test_mlx_package_fails_with_install_instructions() -> None:
    result = run_without_mlx("import decoder_mlx")
    assert result.returncode != 0
    assert "pip install 'decoder-from-scratch[mlx]'" in result.stderr

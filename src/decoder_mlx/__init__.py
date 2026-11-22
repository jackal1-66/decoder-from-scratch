"""MLX implementation of the decoder in the ``decoder`` package."""

try:
    import mlx.core  # noqa: F401
except ImportError as err:
    # Also reached when `mlx` is installed without a backend on Linux (libmlx.so missing),
    # so the original error is kept in the message.
    raise ImportError(
        "decoder_mlx needs a working MLX installation; install it with "
        f"pip install 'decoder-from-scratch[mlx]' (import failed: {err})"
    ) from err

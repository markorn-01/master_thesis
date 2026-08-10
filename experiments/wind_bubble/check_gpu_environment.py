"""Verify that JAX can execute work on an allocated NVIDIA GPU."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a CPU backend for testing this script on a laptop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"JAX version     : {jax.__version__}")
    print(f"Default backend : {jax.default_backend()}")
    print(f"Devices         : {jax.devices()}")

    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError(
            "JAX did not detect a GPU. Run inside a Slurm GPU allocation and "
            "install the CUDA-enabled JAX wheel."
        )

    size = 2048 if jax.default_backend() == "gpu" else 256
    matrix = jnp.ones((size, size), dtype=jnp.float32)
    start = time.perf_counter()
    result = (matrix @ matrix).block_until_ready()
    elapsed = time.perf_counter() - start
    expected = float(size)
    observed = float(result[0, 0])
    if observed != expected:
        raise RuntimeError(f"Incorrect matrix product: {observed} != {expected}")
    print(f"Matrix test     : PASS ({size}x{size}, {elapsed:.3f} s)")


if __name__ == "__main__":
    main()

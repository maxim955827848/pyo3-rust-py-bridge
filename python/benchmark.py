"""
Benchmark: Pure Python vs native Rust (serial) vs native Rust (parallel).

Runs the same Monte-Carlo π estimate three ways and prints a comparison table
with wall-clock times, throughput, and speedup factors. Everything printed is
measured on the machine you run it on — nothing is hard-coded.

    python python/benchmark.py                 # default sample counts
    python python/benchmark.py 20_000_000      # custom sample count
"""
from __future__ import annotations

import math
import os
import sys
import time

import pyo3_rust_py_bridge as native

sys.path.insert(0, os.path.dirname(__file__))
import pure_python  # noqa: E402


def _time(fn, *args) -> tuple[float, float]:
    """Return (result, elapsed_seconds) for a single call."""
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def run(samples: int, seed: int = 42) -> None:
    cores = os.cpu_count() or 1
    print(f"\nMonte-Carlo π  —  samples={samples:,}  seed={seed}  cores={cores}")
    print("-" * 74)

    # Pure Python is the slow baseline; give it a smaller budget on huge runs so
    # the benchmark finishes in reasonable time, then normalise by throughput.
    py_samples = samples if samples <= 5_000_000 else 5_000_000
    py_pi, py_t = _time(pure_python.monte_carlo_pi, py_samples, seed)
    py_rate = py_samples / py_t

    rs_pi, rs_t = _time(native.monte_carlo_pi_serial, samples, seed)
    rs_rate = samples / rs_t

    par_pi, par_t = _time(native.monte_carlo_pi, samples, seed)
    par_rate = samples / par_t

    rows = [
        ("Pure Python (1 thread)", py_pi, py_rate, 1.0),
        ("Rust PyO3 (1 thread)", rs_pi, rs_rate, rs_rate / py_rate),
        (f"Rust PyO3 + rayon ({cores} threads)", par_pi, par_rate, par_rate / py_rate),
    ]

    print(f"{'implementation':<34}{'π estimate':>12}{'M samples/s':>14}{'speedup':>12}")
    print("-" * 74)
    for name, pi, rate, speedup in rows:
        print(f"{name:<34}{pi:>12.5f}{rate / 1e6:>14.2f}{speedup:>11.1f}x")
    print("-" * 74)
    print(f"true π = {math.pi:.5f}   (pure-python timed on {py_samples:,} samples, "
          f"speedups normalised by throughput)")


if __name__ == "__main__":
    n = int(sys.argv[1].replace("_", "")) if len(sys.argv) > 1 else 20_000_000
    run(n)

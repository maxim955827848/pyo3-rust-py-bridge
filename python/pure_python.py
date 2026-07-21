"""
The pure-Python reference implementation.

Deliberately idiomatic, straightforward Python — the code you would actually
write before reaching for a native extension. It mirrors the Rust algorithm
exactly (same SplitMix64 PRNG, same draw order), so for a given seed the two
produce the *same* estimate and the benchmark compares like with like.

This runs single-threaded. That is not a handicap we imposed: for CPU-bound
Python, the GIL serialises threads anyway, so `threading` would not speed this
up. Bypassing that ceiling is exactly what the Rust extension demonstrates.
"""
from __future__ import annotations

_MASK = (1 << 64) - 1


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    z = z ^ (z >> 31)
    return z, state


def monte_carlo_pi(samples: int, seed: int = 42) -> float:
    """Estimate π by the fraction of random points landing in the unit circle."""
    state = (seed + 1) & _MASK
    hits = 0
    denom = float(1 << 53)
    for _ in range(samples):
        r, state = _splitmix64(state)
        x = (r >> 11) / denom
        r, state = _splitmix64(state)
        y = (r >> 11) / denom
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / samples

# pyo3-rust-py-bridge

[![CI](https://github.com/maxim955827848/pyo3-rust-py-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/maxim955827848/pyo3-rust-py-bridge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-stable-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Call native, multi-threaded Rust from Python — and leave the GIL behind for CPU-bound work.**

> **About this project.** A standalone, generalized extraction of an architecture pattern from
> [Ventute](https://ventute.com) — a production AI-driven business-simulation platform — distilled
> into self-contained, runnable form. Published as a portfolio piece demonstrating high-performance
> Python/Rust interop and GIL-free parallel compute.
> Author: [@maxim955827848](https://github.com/maxim955827848).

A minimal, domain-free showcase of the [PyO3](https://pyo3.rs) + [rayon](https://docs.rs/rayon)
pattern: a tight numeric loop written in Rust, exposed as an ordinary Python
function, that releases Python's Global Interpreter Lock and fans the work across
every core. The example task is a Monte-Carlo estimate of π — embarrassingly
parallel, purely arithmetic, tiny data in and out.

---

## Benchmark

Estimating π from **20,000,000** random samples, three ways, running the
*identical* algorithm (same SplitMix64 PRNG, same draw order) so the comparison
is like-for-like:

| Implementation | π estimate | Throughput (M samples/s) | Speedup |
|---|---:|---:|---:|
| Pure Python (1 thread) | 3.14179 | 0.20 | 1.0× |
| Rust PyO3 (1 thread) | 3.14163 | 69.0 | **341×** |
| Rust PyO3 + rayon (8 threads) | 3.14140 | 210.0 | **1039×** |

<sub>Measured on an 8-core machine, release build (`opt-level=3`, LTO). Your
numbers will vary with CPU and core count — reproduce them with
`python python/benchmark.py`. The two axes are independent: **~341×** is the
language/compilation win (native code vs the interpreter), and the further jump
to **~1039×** is the multi-core win the GIL denies to pure-Python threads.</sub>

Two honest caveats so the number means what it says:

- The baseline is a **naive pure-Python loop**, not NumPy. That is the right
  comparison here because both sides run the same scalar algorithm — it isolates
  the cost of the interpreter itself. A vectorised NumPy baseline would be far
  faster (and a different lesson).
- Most of the serial win is Python's **per-operation interpreter overhead** on a
  hot scalar loop; the parallel win on top is pure hardware the GIL otherwise
  leaves idle.

---

## How it bypasses the GIL

Python's **Global Interpreter Lock** lets only one thread execute Python bytecode
at a time. For I/O you can work around it; for **CPU-bound** work it is a hard
ceiling — spawning four Python threads to crunch numbers gets you roughly *one*
core's worth of throughput, because they take turns holding the lock.

A native extension sidesteps this in two steps:

1. **The heavy loop holds no Python objects.** Once we've read the two integer
   arguments, the Monte-Carlo loop is pure Rust `f64` arithmetic. There is no
   reason to hold the GIL while it runs.

2. **`py.allow_threads(...)` releases the GIL for the duration of the compute.**
   Inside that closure the Rust code is free of the interpreter entirely, so
   [`rayon`](https://docs.rs/rayon) can split the samples across a thread per core
   and run them **truly in parallel**. The GIL is re-acquired automatically when
   the closure returns to hand the result back to Python.

```rust
#[pyfunction]
fn monte_carlo_pi(py: Python<'_>, samples: u64, seed: u64) -> f64 {
    // GIL released here → rayon runs on every core with zero interpreter contention.
    let hits = py.allow_threads(|| total_hits_parallel(samples, seed));
    4.0 * hits as f64 / samples as f64   // GIL re-acquired to return to Python.
}
```

Determinism is preserved: the work is split into a fixed number of chunks, each
seeded independently, and the per-chunk hit counts are summed — so the estimate
is identical for a given seed no matter how many threads rayon happens to use.

```mermaid
flowchart LR
    PY["Python caller"] -->|"samples, seed (int)"| FFI["#[pyfunction]<br/>monte_carlo_pi"]
    FFI --> REL["py.allow_threads()<br/>— GIL released —"]
    REL --> R["rayon par_iter<br/>256 chunks over N cores"]
    R --> SUM["Σ hits (order-independent)"]
    SUM -->|"GIL re-acquired"| OUT["float π → Python"]
```

---

## Project layout

```
pyo3-rust-py-bridge/
├── Cargo.toml            # Rust crate: pyo3 (extension-module) + rayon, LTO release profile
├── src/lib.rs            # the native module: serial + parallel Monte-Carlo π
├── pyproject.toml        # maturin build backend
├── python/
│   ├── pure_python.py    # naive Python reference (identical algorithm)
│   └── benchmark.py      # measures & prints the comparison table
├── tests/test_bridge.py  # correctness + determinism + Python/Rust parity
└── .github/workflows/ci.yml
```

---

## Build & run

Requires **Python 3.11+** and a **Rust toolchain** (`rustup`).

```bash
git clone https://github.com/maxim955827848/pyo3-rust-py-bridge.git
cd pyo3-rust-py-bridge

python -m venv .venv && source .venv/bin/activate
pip install maturin pytest

# Compile the Rust crate and install it into the venv as an importable module.
# --release is essential: a debug build is many times slower and understates Rust.
maturin develop --release

python python/benchmark.py          # print the comparison table (real numbers)
pytest                              # correctness + determinism
```

Then it's just a normal Python import:

```python
import pyo3_rust_py_bridge as fast

pi = fast.monte_carlo_pi(50_000_000, seed=42)   # parallel, GIL-free
print(pi)                                        # ≈ 3.14159
```

> **Building on Python 3.14+?** PyO3 0.22 targets CPython ≤ 3.13. To compile
> against a newer interpreter, either build under 3.11/3.12 or set
> `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `maturin develop`.

### Available functions

| Function | Description |
|---|---|
| `monte_carlo_pi(samples, seed=42)` | Parallel (rayon, GIL-free) estimate of π. |
| `monte_carlo_pi_serial(samples, seed=42)` | Single-threaded native estimate (isolates the language speedup). |
| `estimate_pi_samples(samples, seed=42)` | `(estimate, hit_count)` — for tests/inspection. |

## License

MIT — see [LICENSE](LICENSE).

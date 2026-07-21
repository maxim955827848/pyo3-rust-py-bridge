//! A minimal, domain-free PyO3 bridge demonstrating the pattern that matters for
//! CPU-bound work: **do the heavy loop in native code, release the GIL, and let
//! `rayon` fan it across every core.**
//!
//! The example task is a Monte-Carlo estimate of π — embarrassingly parallel,
//! purely numeric, tiny FFI surface (two integers in, one float out). It is the
//! right *shape* for a native extension: almost all the time is spent in a hot
//! arithmetic loop with no Python objects involved, so the interpreter has
//! nothing to do while it runs.
//!
//! Three entry points let a benchmark separate the two independent wins:
//!   * `monte_carlo_pi_serial`  — native, single-threaded (language speedup only).
//!   * `monte_carlo_pi`         — native, `rayon`-parallel, GIL released (adds the
//!                                multi-core speedup Python threads cannot get for
//!                                CPU-bound code).
//!   * `estimate_pi_samples`    — convenience returning both the estimate and the
//!                                hit count, for correctness tests.

use pyo3::prelude::*;
use rayon::prelude::*;

// ── Deterministic PRNG (SplitMix64) ──────────────────────────────────────────
// A tiny, fast, seedable generator so runs are reproducible and each parallel
// chunk gets an independent, fixed stream — the estimate is identical regardless
// of how many threads rayon happens to use.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    // 53-bit mantissa uniform in [0, 1).
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Count how many of `samples` random points fall inside the unit quarter-circle.
/// Pure, allocation-free hot loop — this is the work that must be fast.
#[inline]
fn count_hits(samples: u64, seed: u64) -> u64 {
    let mut state = seed.wrapping_add(1);
    let mut hits = 0u64;
    for _ in 0..samples {
        let x = next_unit(&mut state);
        let y = next_unit(&mut state);
        if x * x + y * y <= 1.0 {
            hits += 1;
        }
    }
    hits
}

// Number of independent chunks the parallel version splits work into. More than
// the core count so rayon's work-stealing keeps every thread busy, but fixed so
// the result is deterministic for a given seed.
const CHUNKS: u64 = 256;

fn total_hits_parallel(samples: u64, seed: u64) -> u64 {
    let per = samples / CHUNKS;
    let rem = samples % CHUNKS;
    (0..CHUNKS)
        .into_par_iter()
        .map(|c| {
            // Each chunk: an independent, fixed seed stream (order-independent sum).
            let n = per + if c < rem { 1 } else { 0 };
            let chunk_seed = seed
                .wrapping_add(c.wrapping_mul(0x2545_F491_4F6C_DD1D))
                .wrapping_add(1);
            count_hits(n, chunk_seed)
        })
        .sum()
}

/// monte_carlo_pi(samples, seed=42) -> float
///
/// Parallel estimate. `py.allow_threads` releases the GIL for the whole compute
/// so the rayon pool runs on every core with zero interpreter contention.
#[pyfunction]
#[pyo3(signature = (samples, seed = 42))]
fn monte_carlo_pi(py: Python<'_>, samples: u64, seed: u64) -> f64 {
    let hits = py.allow_threads(|| total_hits_parallel(samples, seed));
    4.0 * hits as f64 / samples as f64
}

/// monte_carlo_pi_serial(samples, seed=42) -> float
///
/// Single-threaded native estimate. Isolates the language speedup from the
/// parallel speedup when benchmarking.
#[pyfunction]
#[pyo3(signature = (samples, seed = 42))]
fn monte_carlo_pi_serial(py: Python<'_>, samples: u64, seed: u64) -> f64 {
    let hits = py.allow_threads(|| count_hits(samples, seed));
    4.0 * hits as f64 / samples as f64
}

/// estimate_pi_samples(samples, seed=42) -> (estimate, hits)
///
/// Parallel estimate returning the raw hit count too — handy for tests that
/// assert determinism and the estimate/hit relationship.
#[pyfunction]
#[pyo3(signature = (samples, seed = 42))]
fn estimate_pi_samples(py: Python<'_>, samples: u64, seed: u64) -> (f64, u64) {
    let hits = py.allow_threads(|| total_hits_parallel(samples, seed));
    (4.0 * hits as f64 / samples as f64, hits)
}

#[pymodule]
fn pyo3_rust_py_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(monte_carlo_pi, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_pi_serial, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_pi_samples, m)?)?;
    m.add("__doc__", "PyO3 + rayon bridge: GIL-free parallel Monte-Carlo π.")?;
    Ok(())
}

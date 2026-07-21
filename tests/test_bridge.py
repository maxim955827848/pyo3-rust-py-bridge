"""Correctness and determinism of the native bridge."""
import math
import os
import sys

import pyo3_rust_py_bridge as native

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import pure_python  # noqa: E402


def test_estimate_is_close_to_pi():
    pi = native.monte_carlo_pi(2_000_000, 42)
    assert math.isclose(pi, math.pi, abs_tol=0.01)


def test_serial_and_parallel_agree_exactly_is_not_required_but_both_accurate():
    # Serial and parallel use different chunking, so they need not be bit-equal,
    # but both must converge on π.
    serial = native.monte_carlo_pi_serial(2_000_000, 7)
    parallel = native.monte_carlo_pi(2_000_000, 7)
    assert math.isclose(serial, math.pi, abs_tol=0.01)
    assert math.isclose(parallel, math.pi, abs_tol=0.01)


def test_parallel_is_deterministic_across_calls():
    # Same seed + sample count → identical estimate every time, regardless of how
    # rayon schedules the chunks.
    a = native.monte_carlo_pi(1_000_000, 123)
    b = native.monte_carlo_pi(1_000_000, 123)
    assert a == b


def test_estimate_pi_samples_returns_consistent_hits():
    pi, hits = native.estimate_pi_samples(1_000_000, 42)
    assert hits > 0
    assert math.isclose(pi, 4.0 * hits / 1_000_000)


def test_serial_native_matches_pure_python_bit_for_bit():
    # The Rust single-threaded path and the pure-Python reference share the exact
    # PRNG and draw order, so for one seed they must produce the identical float.
    n, seed = 200_000, 999
    assert native.monte_carlo_pi_serial(n, seed) == pure_python.monte_carlo_pi(n, seed)


def test_different_seeds_give_different_estimates():
    assert native.monte_carlo_pi(500_000, 1) != native.monte_carlo_pi(500_000, 2)

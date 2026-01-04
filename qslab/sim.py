from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GBMParams:
    """Geometric Brownian Motion parameters."""
    s0: float
    mu: float
    sigma: float
    dt: float
    steps: int
    n_paths: int
    seed: int | None = 42


def simulate_gbm(p: GBMParams) -> np.ndarray:
    """
    Simulate GBM price paths.

    Returns:
        prices: shape (n_paths, steps+1)
    """
    if p.s0 <= 0:
        raise ValueError("s0 must be > 0")
    if p.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if p.dt <= 0:
        raise ValueError("dt must be > 0")
    if p.steps <= 0 or p.n_paths <= 0:
        raise ValueError("steps and n_paths must be > 0")

    rng = np.random.default_rng(p.seed)
    z = rng.standard_normal(size=(p.n_paths, p.steps))

    drift = (p.mu - 0.5 * p.sigma**2) * p.dt
    diffusion = p.sigma * np.sqrt(p.dt) * z
    log_returns = drift + diffusion

    prices = np.empty((p.n_paths, p.steps + 1), dtype=float)
    prices[:, 0] = p.s0
    prices[:, 1:] = p.s0 * np.exp(np.cumsum(log_returns, axis=1))
    return prices


def _self_test() -> None:
    params = GBMParams(
        s0=100.0,
        mu=0.10,
        sigma=0.20,
        dt=1 / 252,
        steps=252,
        n_paths=3,
        seed=123,
    )

    prices = simulate_gbm(params)

    assert prices.shape == (3, 253)
    assert np.all(prices[:, 0] == 100.0)
    assert np.all(prices > 0)

    print("OK: GBM simulation works.")
    print("First path (first 5 prices):", np.round(prices[0, :5], 4))


if __name__ == "__main__":
    _self_test()

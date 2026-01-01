from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    returns: np.ndarray
    pnl: float
    total_return: float


def buy_and_hold(prices: np.ndarray, initial_cash: float = 10_000.0) -> BacktestResult:
    """
    Simplest possible strategy:
    - Buy 1 share at t=0
    - Hold until the end
    """
    if prices.ndim != 1:
        raise ValueError("prices must be 1D")
    if len(prices) < 2:
        raise ValueError("prices must have at least 2 points")

    shares = 1.0
    cash = initial_cash - prices[0]

    equity = cash + shares * prices
    returns = equity[1:] / equity[:-1] - 1.0
    pnl = equity[-1] - equity[0]
    total_return = equity[-1] / equity[0] - 1.0

    return BacktestResult(equity=equity, returns=returns, pnl=pnl, total_return=total_return)


def _self_test() -> None:
    prices = np.array([100.0, 102.0, 101.0, 105.0])
    res = buy_and_hold(prices)

    print("OK: backtest works")
    print("Equity:", np.round(res.equity, 2))
    print("PnL:", round(res.pnl, 2))
    print("Total return:", round(res.total_return * 100, 2), "%")


if __name__ == "__main__":
    _self_test()

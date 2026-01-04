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

    return BacktestResult(
        equity=equity,
        returns=returns,
        pnl=pnl,
        total_return=total_return,
    )

def _sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average. Returns an array same length as prices.
    First (window-1) values are NaN because SMA isn't defined yet.
    """
    prices = np.asarray(prices, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(prices):
        # no SMA possible anywhere
        return np.full_like(prices, np.nan, dtype=float)

    sma = np.full_like(prices, np.nan, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    sma_valid = np.convolve(prices, kernel, mode="valid")  # length = n - window + 1
    sma[window - 1:] = sma_valid
    return sma


def moving_average_crossover(
    prices: np.ndarray,
    short_window: int = 20,
    long_window: int = 50,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """
    MA crossover:
      - position(t) = 1 if SMA_short(t) > SMA_long(t), else 0
      - portfolio_return(t->t+1) = position(t) * price_return(t->t+1)

    Returns BacktestResult with equity aligned to prices length.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be 1D")
    if len(prices) < 2:
        equity = np.array([initial_capital], dtype=float)
        returns = np.array([], dtype=float)
        return BacktestResult(equity=equity, returns=returns, pnl=0.0, total_return=0.0)

    if short_window >= long_window:
        raise ValueError("short_window must be < long_window")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")

    # 1) compute price returns (length n-1)
    price_rets = prices[1:] / prices[:-1] - 1.0

    # 2) compute moving averages (length n)
    sma_s = _sma(prices, short_window)
    sma_l = _sma(prices, long_window)

    # 3) signal/position at time t (length n)
    #    if either SMA is NaN (warmup), position = 0 (stay in cash)
    position = np.where((sma_s > sma_l), 1.0, 0.0)
    position = np.where(np.isnan(sma_s) | np.isnan(sma_l), 0.0, position)

    # 4) portfolio returns use position at t applied to price return t->t+1
    strat_rets = position[:-1] * price_rets   # length n-1

    # 5) build equity curve (length n)
    equity = np.empty(len(prices), dtype=float)
    equity[0] = initial_capital
    equity[1:] = initial_capital * np.cumprod(1.0 + strat_rets)

    pnl = float(equity[-1] - initial_capital)
    total_return = float(equity[-1] / initial_capital - 1.0)

    return BacktestResult(equity=equity, returns=strat_rets, pnl=pnl, total_return=total_return)


def _self_test() -> None:
    prices = np.array([100.0, 102.0, 101.0, 105.0])
    res = buy_and_hold(prices)

    print("OK: backtest works")
    print("Equity:", np.round(res.equity, 2))
    print("PnL:", round(res.pnl, 2))
    print("Total return:", round(res.total_return * 100, 2), "%")


if __name__ == "__main__":
    _self_test()

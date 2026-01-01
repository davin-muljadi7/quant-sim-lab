from qslab.sim import GBMParams, simulate_gbm
from qslab.backtest import buy_and_hold
from qslab.metrics import max_drawdown


import numpy as np


def main() -> None:
    # 1) Simulate ONE GBM price path
    params = GBMParams(
        s0=100.0,
        mu=0.08,
        sigma=0.20,
        dt=1 / 252,
        steps=252,
        n_paths=1000,
        seed=42,
    )

    prices = simulate_gbm(params)  # shape: (n_paths, steps+1)
    pnls = np.empty(params.n_paths, dtype=float)
    total_returns = np.empty(params.n_paths, dtype=float)
    mdds = np.empty(params.n_paths, dtype=float)

    for i in range(params.n_paths):
        path = prices[i]
        res = buy_and_hold(path)

        pnls[i] = res.pnl
        total_returns[i] = res.total_return
        mdds[i] = max_drawdown(res.equity)


    print("=== Monte Carlo Summary (Buy & Hold on GBM) ===")
    print("Paths:", params.n_paths)
    print("Mean PnL:", round(float(pnls.mean()), 2))
    print("Median PnL:", round(float(np.median(pnls)), 2))
    print("Best PnL:", round(float(pnls.max()), 2))
    print("Worst PnL:", round(float(pnls.min()), 2))
    print("% Profitable:", round(float((pnls > 0).mean() * 100), 2), "%")
    print("Mean total return:", round(float(total_returns.mean() * 100), 2), "%")
    print("Mean max drawdown:", round(float(mdds.mean() * 100), 2), "%")
    print("Worst max drawdown:", round(float(mdds.max() * 100), 2), "%")



if __name__ == "__main__":
    main()

from qslab.sim import GBMParams, simulate_gbm
from qslab.metrics import max_drawdown
from qslab.backtest import buy_and_hold, moving_average_crossover



import numpy as np

def print_summary(
        pnls: np.ndarray,
        total_returns: np.ndarray,
        mdds: np.ndarray
) -> None:

    print("=== Monte Carlo Summary (Buy & Hold on GBM) ===")
    print("Paths:", len(pnls))
    print("Mean PnL:", round(float(pnls.mean()), 2))
    print("Median PnL:", round(float(np.median(pnls)), 2))
    print("Best PnL:", round(float(pnls.max()), 2))
    print("Worst PnL:", round(float(pnls.min()), 2))
    print("% Profitable:", round(float((pnls > 0).mean() * 100), 2), "%")
    print("Mean total return:", round(float(total_returns.mean() * 100), 2), "%")
    print("Mean max drawdown:", round(float(mdds.mean() * 100), 2), "%")
    print("Worst max drawdown:", round(float(mdds.max() * 100), 2), "%")

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

    # Buy and Hold storage
    pnls_bh = np.empty(params.n_paths, dtype=float)
    total_returns_bh = np.empty(params.n_paths, dtype=float)
    mdds_bh = np.empty(params.n_paths, dtype=float)

    # MA Crossover storage
    pnls_ma = np.empty(params.n_paths, dtype=float)
    total_returns_ma = np.empty(params.n_paths, dtype=float)
    mdds_ma = np.empty(params.n_paths, dtype=float)

    for i in range(params.n_paths):
        path = prices[i]

        # --- Buy & Hold ---
        res_bh = buy_and_hold(path)
        pnls_bh[i] = res_bh.pnl
        total_returns_bh[i] = res_bh.total_return
        mdds_bh[i] = max_drawdown(res_bh.equity)

        # --- Moving Average Crossover ---
        res_ma = moving_average_crossover(path, short_window=20, long_window=50)
        pnls_ma[i] = res_ma.pnl
        total_returns_ma[i] = res_ma.total_return
        mdds_ma[i] = max_drawdown(res_ma.equity)

    print_summary("Buy & Hold on GBM", pnls_bh, total_returns_bh, mdds_bh)
    print_summary("MA Crossover (20/50) on GBM", pnls_ma, total_returns_ma, mdds_ma)



if __name__ == "__main__":
    main()

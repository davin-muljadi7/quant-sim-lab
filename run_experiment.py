from qslab.sim import GBMParams, simulate_gbm
from qslab.backtest import buy_and_hold, moving_average_crossover
from qslab.db import init_db, insert_result
from qslab.metrics import max_drawdown, volatility, sharpe_ratio



import numpy as np

def print_summary(
        title: str,
        pnls: np.ndarray,
        total_returns: np.ndarray,
        mdds: np.ndarray
) -> None:

    print(f"=== Monte Carlo Summary ({title}) ===")
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
    init_db()

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

    # Buy and Hold arrays 
    pnls_bh = np.empty(params.n_paths)
    total_returns_bh = np.empty(params.n_paths)
    mdds_bh = np.empty(params.n_paths)
    vols_bh = np.empty(params.n_paths)
    sharpes_bh = np.empty(params.n_paths)

    # MA Crossover arrays
    pnls_ma = np.empty(params.n_paths)
    total_returns_ma = np.empty(params.n_paths)
    mdds_ma = np.empty(params.n_paths)
    vols_ma = np.empty(params.n_paths)
    sharpes_ma = np.empty(params.n_paths)


    # Buy and Hold storage
    bh_mean_pnl = float(pnls_bh.mean())
    bh_mean_return = float(total_returns_bh.mean())
    bh_mean_dd = float(mdds_bh.mean())
    bh_worst_dd = float(mdds_bh.max())
    bh_mean_vol = float(vols_bh.mean())
    bh_mean_sharpe = float(sharpes_bh.mean())

    insert_result(
        strategy="buy_and_hold",
        n_paths=params.n_paths,
        mean_pnl=bh_mean_pnl,
        mean_return=bh_mean_return,
        mean_drawdown=bh_mean_dd,
        worst_drawdown=bh_worst_dd,
        mean_volatility=bh_mean_vol,
        mean_sharpe=bh_mean_sharpe,
    )

    # MA Crossover storage
    ma_mean_pnl = float(pnls_ma.mean())
    ma_mean_return = float(total_returns_ma.mean())
    ma_mean_dd = float(mdds_ma.mean())
    ma_worst_dd = float(mdds_ma.max())
    ma_mean_vol = float(vols_ma.mean())
    ma_mean_sharpe = float(sharpes_ma.mean())

    insert_result(
        strategy="ma_crossover_20_50",
        n_paths=params.n_paths,
        mean_pnl=ma_mean_pnl,
        mean_return=ma_mean_return,
        mean_drawdown=ma_mean_dd,
        worst_drawdown=ma_worst_dd,
        mean_volatility=ma_mean_vol,
        mean_sharpe=ma_mean_sharpe,
    )

    for i in range(params.n_paths):
        path = prices[i]

        # Buy and Hold
        res_bh = buy_and_hold(path)
        pnls_bh[i] = res_bh.pnl
        total_returns_bh[i] = res_bh.total_return
        mdds_bh[i] = max_drawdown(res_bh.equity)

        # MA Crossover 
        res_ma = moving_average_crossover(path, short_window=20, long_window=50)
        pnls_ma[i] = res_ma.pnl
        total_returns_ma[i] = res_ma.total_return
        mdds_ma[i] = max_drawdown(res_ma.equity)

    print_summary("Buy & Hold on GBM", pnls_bh, total_returns_bh, mdds_bh)
    print_summary("MA Crossover (20/50) on GBM", pnls_ma, total_returns_ma, mdds_ma)



if __name__ == "__main__":
    main()

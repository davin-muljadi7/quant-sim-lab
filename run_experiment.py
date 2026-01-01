print("RUN_EXPERIMENT STARTED")
from qslab.sim import GBMParams, simulate_gbm
from qslab.backtest import buy_and_hold

import numpy as np


def main() -> None:
    # 1) Simulate ONE GBM price path
    params = GBMParams(
        s0=100.0,
        mu=0.08,
        sigma=0.20,
        dt=1 / 252,
        steps=252,
        n_paths=1,
        seed=42,
    )

    prices = simulate_gbm(params)
    price_path = prices[0]  # shape (253,)

    # 2) Run backtest on that path
    result = buy_and_hold(price_path)

    # 3) Print results
    print("=== Experiment Result ===")
    print("Final price:", round(price_path[-1], 2))
    print("PnL:", round(result.pnl, 2))
    print("Total return:", round(result.total_return * 100, 2), "%")


if __name__ == "__main__":
    main()

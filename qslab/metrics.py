import numpy as np

def max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum drawdown as a positive fraction (e.g. 0.25 = 25%).
    """
    equity = np.asarray(equity, dtype=float)

    if equity.ndim != 1:
        raise ValueError("equity must be 1D")
    if len(equity) < 2:
        return 0.0

    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0   # ≤ 0
    return float(-drawdowns.min())

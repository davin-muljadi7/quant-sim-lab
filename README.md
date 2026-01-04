# Quant Simulation Lab

An end-to-end Python project that simulates financial price paths, backtests
simple trading strategies, and analyzes their performance using Monte Carlo
simulation and SQLite.

---

## Project Overview

This project explores how different trading strategies behave under uncertainty
by simulating thousands of price paths using **Geometric Brownian Motion (GBM)**.

Each strategy is backtested consistently across all simulated paths, and both
performance and risk metrics are computed. Results are aggregated using
**Monte Carlo simulation** and stored in an **SQLite database** for comparison
using SQL queries.

The focus of this project is correctness, intuition, and clean analytical
workflow — not live trading or machine learning.

---

## Project Structure

- `sim.py`  
  Simulates price paths using Geometric Brownian Motion (GBM)

- `backtest.py`  
  Implements trading strategies:
  - Buy & Hold
  - Moving Average Crossover (20 / 50)

- `metrics.py`  
  Computes performance and risk metrics:
  - Maximum drawdown
  - Volatility
  - Sharpe ratio

- `run_experiment.py`  
  Runs Monte Carlo simulations, aggregates results, and stores summaries

- `db.py`  
  Initializes and writes results to an SQLite database

- `query_results.py`  
  Queries and compares strategies using SQL

---

## Metrics Used

- **PnL**  
  Final profit or loss of the strategy

- **Total return**  
  Percentage change in equity from start to end

- **Maximum drawdown**  
  Largest peak-to-trough loss experienced during the strategy

- **Volatility**  
  Standard deviation of returns, measuring risk

- **Sharpe ratio**  
  Risk-adjusted return (return per unit of volatility)

These metrics allow strategies to be compared not only by returns, but also by
the risks taken to achieve them.

---

## How to Run

1. Initialize the database:
```bash
python qslab/db.py

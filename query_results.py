import sqlite3


def main() -> None:
    conn = sqlite3.connect("results.db")
    cur = conn.cursor()

    print("\n=== Last 10 runs ===")
    rows = cur.execute("""
        SELECT id, strategy, n_paths, mean_return, mean_drawdown, mean_sharpe
        FROM results
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(r)

    print("\n=== Average metrics by strategy ===")
    rows = cur.execute("""
        SELECT
            strategy,
            COUNT(*) AS runs,
            AVG(mean_return) AS avg_return,
            AVG(mean_drawdown) AS avg_dd,
            AVG(mean_sharpe) AS avg_sharpe,
            MIN(mean_sharpe) AS worst_sharpe,
            MAX(mean_sharpe) AS best_sharpe
        FROM results
        GROUP BY strategy
        ORDER BY avg_sharpe DESC
    """).fetchall()

    for r in rows:
        print(r)

    conn.close()


if __name__ == "__main__":
    main()

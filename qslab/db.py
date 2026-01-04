import sqlite3


def init_db(db_path: str = "results.db") -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy TEXT,
        n_paths INTEGER,
        mean_pnl REAL,
        mean_return REAL,
        mean_drawdown REAL,
        worst_drawdown REAL,
        mean_volatility REAL,
        mean_sharpe REAL
    )
    """)

    conn.commit()
    conn.close()


def insert_result(
    strategy: str,
    n_paths: int,
    mean_pnl: float,
    mean_return: float,
    mean_drawdown: float,
    worst_drawdown: float,
    mean_volatility: float,
    mean_sharpe: float,
    db_path: str = "results.db",
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO results (
            strategy, n_paths, mean_pnl, mean_return, mean_drawdown,
            worst_drawdown, mean_volatility, mean_sharpe
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            strategy,
            n_paths,
            mean_pnl,
            mean_return,
            mean_drawdown,
            worst_drawdown,
            mean_volatility,
            mean_sharpe,
        ),
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")

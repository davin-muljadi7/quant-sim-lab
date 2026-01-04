import sqlite3


def init_db(db_path: str = "results.db") -> None:
    """
    Create the SQLite database and results table if they don't exist.
    """
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


if __name__ == "__main__":
    init_db()
    print("Database initialized.")

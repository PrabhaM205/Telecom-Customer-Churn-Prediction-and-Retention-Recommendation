import sqlite3
from pathlib import Path
import pandas as pd


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATABASE_DIR = BASE_DIR / "data" / "monitoring"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = DATABASE_DIR / "monitoring.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DATABASE_PATH
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_dataframe(
    df,
    table_name
):

    if df.empty:
        raise ValueError(
            "Cannot save an empty dataframe."
        )

    connection = get_connection()

    try:

        df.to_sql(
            table_name,
            connection,
            if_exists="append",
            index=False
        )

    finally:

        connection.close()


# ============================================================
# LOAD DATA
# ============================================================

def load_dataframe(
    table_name
):

    connection = get_connection()

    try:

        return pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            connection
        )

    finally:

        connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            feature TEXT,
            psi REAL,
            status TEXT
        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE DRIFT EVENT
# ============================================================

def save_drift_event(
    feature,
    psi,
    status
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO drift_events
        (
            feature,
            psi,
            status
        )
        VALUES (?, ?, ?)
        """,
        (
            feature,
            float(psi),
            status
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print("=" * 60)
    print("MONITORING DATABASE INITIALIZED")
    print("=" * 60)

    print(
        f"Database location:\n{DATABASE_PATH}"
    )
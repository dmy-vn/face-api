import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path("data/faces.db")
THRESHOLD = 0.45          # cosine similarity required to accept a match


def _connect() -> sqlite3.Connection:
    """Open the DB and make sure the table exists (idempotent)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS faces (
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               name      TEXT NOT NULL,
               embedding BLOB NOT NULL
           )"""
    )
    return conn


def save(name: str, embedding: np.ndarray) -> None:
    """Insert one face. Vector is stored as raw float32 bytes (2 KB)."""
    blob = embedding.astype(np.float32).tobytes()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)",
            (name, blob),
        )


def match(embedding: np.ndarray) -> tuple[str, float] | None:
    """Return (name, score) of the best cosine match above THRESHOLD, else None."""
    with _connect() as conn:
        rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
    if not rows:
        return None

    names = [r[0] for r in rows]
    db = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    # cosine similarity for L2-normalized vectors == dot product
    q = embedding / np.linalg.norm(embedding)
    db_n = db / np.linalg.norm(db, axis=1, keepdims=True)
    scores = db_n @ q

    i = int(np.argmax(scores))
    if scores[i] >= THRESHOLD:
        return names[i], float(scores[i])
    return None

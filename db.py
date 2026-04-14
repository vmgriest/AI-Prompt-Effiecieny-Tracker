import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tracker.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text           TEXT    NOT NULL,
            response_text         TEXT,
            model                 TEXT    NOT NULL,
            input_tokens          INTEGER,
            output_tokens         INTEGER,
            total_duration_ms     REAL,
            generation_duration_ms REAL,
            tokens_per_second     REAL,
            quality_score         REAL,
            relevance_score       REAL,
            accuracy_score        REAL,
            completeness_score    REAL,
            conciseness_score     REAL,
            hallucination_detected INTEGER DEFAULT 0,
            judge_model           TEXT,
            tags                  TEXT,
            timestamp             TEXT    NOT NULL,
            improvement_tips      TEXT,
            improved_prompt       TEXT
        )
    """)
    # Migrate existing DBs that predate these columns
    for col in ("improvement_tips", "improved_prompt"):
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def insert_run(data: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (
            prompt_text, response_text, model,
            input_tokens, output_tokens,
            total_duration_ms, generation_duration_ms, tokens_per_second,
            quality_score, relevance_score, accuracy_score,
            completeness_score, conciseness_score, hallucination_detected,
            judge_model, tags, timestamp,
            improvement_tips, improved_prompt
        ) VALUES (
            :prompt_text, :response_text, :model,
            :input_tokens, :output_tokens,
            :total_duration_ms, :generation_duration_ms, :tokens_per_second,
            :quality_score, :relevance_score, :accuracy_score,
            :completeness_score, :conciseness_score, :hallucination_detected,
            :judge_model, :tags, :timestamp,
            :improvement_tips, :improved_prompt
        )
    """, data)
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def get_all_runs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_runs_by_prompt(prompt_text: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs WHERE prompt_text = ? ORDER BY timestamp",
        (prompt_text,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_run_tips(run_id: int, tips: list[str], improved_prompt: str):
    """Persist improvement tips and the rewritten prompt for a run."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE runs SET improvement_tips = ?, improved_prompt = ? WHERE id = ?",
        ("\n".join(tips), improved_prompt, run_id),
    )
    conn.commit()
    conn.close()


def delete_run(run_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()

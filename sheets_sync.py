"""Google Sheets sync for the Prompt Efficiency Tracker.

Setup (one-time):
  1. Go to https://console.cloud.google.com and create a project.
  2. Enable the Google Sheets API for that project.
  3. Create a Service Account (IAM → Service Accounts → Create).
  4. Generate a JSON key for the service account and save it as
     credentials.json in this project directory.
  5. Open your Google Sheet, click Share, and add the service account
     email (looks like xxx@yyy.iam.gserviceaccount.com) as an Editor.
  6. Copy the Spreadsheet ID from the URL:
       https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/edit
     and paste it into the app sidebar.

The sync writes two sheets:
  Prompts — one row per run, all columns
  Models  — aggregated stats per model
"""

import json
from pathlib import Path

import pandas as pd

CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"
CONFIG_PATH = Path(__file__).parent / "sheets_config.json"

# ── config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"spreadsheet_id": "", "auto_sync": False}


def save_config(spreadsheet_id: str, auto_sync: bool):
    CONFIG_PATH.write_text(
        json.dumps({"spreadsheet_id": spreadsheet_id, "auto_sync": auto_sync}, indent=2)
    )


def is_configured() -> tuple[bool, str]:
    """Return (ok, message). ok=True means we can attempt a sync."""
    if not CREDENTIALS_PATH.exists():
        return False, "credentials.json not found in project folder."
    cfg = load_config()
    if not cfg.get("spreadsheet_id", "").strip():
        return False, "No Spreadsheet ID saved yet."
    return True, "OK"


# ── sync ──────────────────────────────────────────────────────────────────────

def sync(runs: list[dict]) -> str:
    """Push all runs to Google Sheets.  Returns a status message."""
    ok, msg = is_configured()
    if not ok:
        return f"Not configured: {msg}"

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return "gspread / google-auth not installed. Run: pip install gspread google-auth"

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        creds = Credentials.from_service_account_file(str(CREDENTIALS_PATH), scopes=scopes)
        gc = gspread.authorize(creds)
    except Exception as exc:
        return f"Auth failed: {exc}"

    cfg = load_config()
    try:
        sh = gc.open_by_key(cfg["spreadsheet_id"])
    except Exception as exc:
        return f"Could not open spreadsheet: {exc}"

    df_all = pd.DataFrame(runs)

    # ── Prompts sheet ─────────────────────────────────────────────────────────
    prompts_cols = [
        "id", "timestamp", "prompt_text", "model",
        "input_tokens", "output_tokens", "tokens_per_second",
        "total_duration_ms", "quality_score",
        "relevance_score", "accuracy_score", "completeness_score", "conciseness_score",
        "hallucination_detected", "judge_model", "tags",
    ]
    df_prompts = df_all[[c for c in prompts_cols if c in df_all.columns]].copy()
    df_prompts.rename(columns={"total_duration_ms": "latency_ms"}, inplace=True)
    df_prompts = df_prompts.fillna("").astype(str)

    # ── Models sheet ──────────────────────────────────────────────────────────
    agg = df_all.groupby("model").agg(
        total_runs         =("id",                     "count"),
        avg_quality        =("quality_score",          "mean"),
        avg_relevance      =("relevance_score",         "mean"),
        avg_accuracy       =("accuracy_score",          "mean"),
        avg_completeness   =("completeness_score",      "mean"),
        avg_conciseness    =("conciseness_score",       "mean"),
        avg_tokens_per_sec =("tokens_per_second",       "mean"),
        avg_latency_ms     =("total_duration_ms",       "mean"),
        avg_input_tokens   =("input_tokens",            "mean"),
        avg_output_tokens  =("output_tokens",           "mean"),
        hallucination_count=("hallucination_detected",  "sum"),
    ).reset_index()
    agg["hallucination_rate_%"] = (
        agg["hallucination_count"] / agg["total_runs"] * 100
    ).round(1)
    for col in agg.select_dtypes("float").columns:
        agg[col] = agg[col].round(2)
    agg = agg.fillna("").astype(str)

    try:
        _write_sheet(sh, "Prompts", df_prompts)
        _write_sheet(sh, "Models", agg)
    except Exception as exc:
        return f"Write failed: {exc}"

    return f"Synced {len(df_prompts)} runs to Google Sheets."


# ── internal ──────────────────────────────────────────────────────────────────

def _write_sheet(spreadsheet, title: str, df: pd.DataFrame):
    """Clear and rewrite a worksheet, creating it if it doesn't exist."""
    try:
        ws = spreadsheet.worksheet(title)
    except Exception:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=30)

    ws.clear()
    header = df.columns.tolist()
    rows = df.values.tolist()
    ws.update([header] + rows, value_input_option="USER_ENTERED")

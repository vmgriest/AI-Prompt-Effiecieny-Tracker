# Prompt Efficiency Tracker

A local Streamlit app for measuring and improving the quality of prompts across local LLMs via [Ollama](https://ollama.com).

Every run is automatically scored by a judge model, improvement tips are generated, and results are stored in SQLite — with optional sync to Google Sheets.

---

## Features

- **Run prompts** against one or more local Ollama models simultaneously
- **A/B test** two prompt variants side by side
- **Automatic evaluation** — a judge LLM scores every response on relevance, accuracy, completeness, and conciseness (1–10)
- **Automatic improvement tips** — the judge suggests how to rewrite the prompt for a better score
- **Dashboard** — quality trends, model comparison, latency vs quality scatter, tokens/sec charts
- **History** — full run log with filters, response inspector, and per-run delete
- **Export** — download all data as a two-sheet Excel file (Prompts + Models)
- **Google Sheets sync** — auto-sync after every run or manually on demand

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with at least one model pulled

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start Ollama**
```bash
ollama serve
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

---

## Models used in development

| Model | Size | Recommended use |
|-------|------|-----------------|
| `llama3:8b` | 4.7 GB | Judge model (fast evaluation) |
| `gemma4:26b` | 17 GB | Generation |
| `glm-4.7-flash:latest` | 19 GB | Generation |

Set the judge model in the sidebar. Use larger models for generation and a smaller one for judging to keep evaluation fast.

---

## Google Sheets sync (optional)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and create a project
2. Enable the **Google Sheets API**
3. Create a **Service Account** → generate a JSON key → save it as `credentials.json` in the project folder
4. Share your Google Sheet with the service account email (give it **Editor** access)
5. Copy the Spreadsheet ID from the sheet URL:
   `https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/edit`
6. Paste the ID into the sidebar in the app → click **Save Sheets config**

The sheet will have two tabs:
- **Prompts** — one row per run including scores, tips, and suggested rewrites
- **Models** — aggregated stats per model

> `credentials.json` and `sheets_config.json` are in `.gitignore` and will never be committed.

---

## Project structure

```
├── app.py              # Streamlit dashboard (4 tabs: Run, A/B Test, Dashboard, History)
├── db.py               # SQLite layer (tracker.db auto-created on first run)
├── ollama_client.py    # HTTP wrapper for Ollama /api/generate (streaming)
├── evaluator.py        # LLM-as-judge scoring + improvement tip generation
├── sheets_sync.py      # Google Sheets sync
├── requirements.txt
└── .gitignore
```

---

## Metrics tracked per run

| Metric | Source |
|--------|--------|
| Input / output tokens | Ollama response body |
| Tokens per second | Derived from `eval_duration` |
| Latency (ms) | Ollama `total_duration` |
| Relevance, Accuracy, Completeness, Conciseness (1–10) | Judge LLM |
| Hallucination detected | Judge LLM |
| Quality score (composite) | Weighted average of above; 70% penalty if hallucination |
| Improvement tips | Judge LLM coaching prompt |
| Suggested rewrite | Judge LLM coaching prompt |

---

## Files not to commit

| File | Why |
|------|-----|
| `credentials.json` | Google service account private key |
| `sheets_config.json` | Contains your Spreadsheet ID |
| `tracker.db` | Local run data |

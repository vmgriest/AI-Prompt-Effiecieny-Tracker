"""Prompt Efficiency Tracker — Streamlit app.

Tabs:
  Run       — fire a prompt at one or more local Ollama models
  A/B Test  — compare two prompt variants side by side
  Dashboard — charts: quality trends, model comparison, efficiency scatter
  History   — full run log with delete
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
import evaluator
import ollama_client

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Efficiency Tracker",
    page_icon="🎯",
    layout="wide",
)

db.init_db()

# ── helpers ───────────────────────────────────────────────────────────────────

def _models() -> list[str]:
    models = ollama_client.list_models()
    return models if models else ["(Ollama not reachable)"]


def _run_and_record(prompt: str, model: str, judge_model: str, tags: str = "") -> dict:
    """Call Ollama, evaluate with judge, persist to DB.  Returns the full record."""
    with st.spinner(f"Generating with **{model}**…"):
        gen = ollama_client.generate(prompt, model)

    with st.spinner(f"Evaluating with judge **{judge_model}**…"):
        ev = evaluator.evaluate(prompt, gen["response"], judge_model)

    record = {
        "prompt_text":             prompt,
        "response_text":           gen["response"],
        "model":                   model,
        "input_tokens":            gen["input_tokens"],
        "output_tokens":           gen["output_tokens"],
        "total_duration_ms":       gen["total_duration_ms"],
        "generation_duration_ms":  gen["generation_duration_ms"],
        "tokens_per_second":       gen["tokens_per_second"],
        "quality_score":           ev.get("quality_score", 5.0),
        "relevance_score":         ev.get("relevance", 5.0),
        "accuracy_score":          ev.get("accuracy", 5.0),
        "completeness_score":      ev.get("completeness", 5.0),
        "conciseness_score":       ev.get("conciseness", 5.0),
        "hallucination_detected":  int(ev.get("hallucination_detected", False)),
        "judge_model":             judge_model,
        "tags":                    tags,
        "timestamp":               datetime.now().isoformat(timespec="seconds"),
    }
    db.insert_run(record)
    return {**record, **gen, **ev}


def _metric_cards(result: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quality", f"{result['quality_score']:.1f} / 10")
    c2.metric("Tokens in / out",
              f"{result['input_tokens']} / {result['output_tokens']}")
    c3.metric("Speed", f"{result['tokens_per_second']:.1f} tok/s")
    c4.metric("Latency", f"{result['total_duration_ms'] / 1000:.2f} s")
    halluc = "⚠ YES" if result["hallucination_detected"] else "✓ No"
    c5.metric("Hallucination", halluc)


def _score_radar(result: dict, label: str):
    dims = ["Relevance", "Accuracy", "Completeness", "Conciseness"]
    vals = [
        result.get("relevance_score",    result.get("relevance",    5)),
        result.get("accuracy_score",     result.get("accuracy",     5)),
        result.get("completeness_score", result.get("completeness", 5)),
        result.get("conciseness_score",  result.get("conciseness",  5)),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=dims + [dims[0]],
        fill="toself",
        name=label,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        title=label,
    )
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎯 Prompt Tracker")

    if not ollama_client.is_running():
        st.error("Ollama is not running.  Start it with `ollama serve`.")
    else:
        st.success("Ollama connected")

    available = _models()

    st.markdown("---")
    st.caption("Judge model (evaluates responses)")
    judge_model = st.selectbox("Judge", available, key="judge_model_select",
                               index=0,
                               help="llama3:8b is fastest for judging.")

    st.markdown("---")
    st.caption("Optional tags (comma-separated)")
    tags_input = st.text_input("Tags", placeholder="experiment-1, zero-shot")


# ── tabs ──────────────────────────────────────────────────────────────────────

tab_run, tab_ab, tab_dash, tab_hist = st.tabs(
    ["▶ Run", "⚖ A/B Test", "📊 Dashboard", "🗂 History"]
)

# ─────────────────────────────────────────────
# TAB 1 — RUN
# ─────────────────────────────────────────────
with tab_run:
    st.header("Run a prompt")

    prompt_input = st.text_area("Prompt", height=120,
                                placeholder="Enter your prompt here…")

    run_models = st.multiselect(
        "Model(s) to test",
        available,
        default=[available[0]],
    )

    if st.button("Run", type="primary", disabled=not prompt_input.strip()):
        if not run_models:
            st.warning("Select at least one model.")
        else:
            results = []
            for model in run_models:
                r = _run_and_record(
                    prompt_input.strip(), model, judge_model, tags_input
                )
                results.append(r)

            for r in results:
                st.subheader(r["model"])
                _metric_cards(r)

                col_resp, col_radar = st.columns([2, 1])
                with col_resp:
                    st.markdown("**Response**")
                    st.markdown(r["response"])
                with col_radar:
                    st.plotly_chart(
                        _score_radar(r, r["model"]),
                        use_container_width=True,
                    )
                st.divider()

            if len(results) > 1:
                st.subheader("Model comparison")
                comp_df = pd.DataFrame([
                    {
                        "Model":      r["model"],
                        "Quality":    r["quality_score"],
                        "Tok/s":      r["tokens_per_second"],
                        "Latency (s)": r["total_duration_ms"] / 1000,
                        "Out tokens": r["output_tokens"],
                    }
                    for r in results
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 2 — A/B TEST
# ─────────────────────────────────────────────
with tab_ab:
    st.header("A/B prompt comparison")
    st.caption("Test two different prompt wordings on the same model.")

    ab_model = st.selectbox("Model", available, key="ab_model")

    col_a, col_b = st.columns(2)
    with col_a:
        prompt_a = st.text_area("Prompt A", height=150, key="pa",
                                placeholder="Version A of your prompt…")
    with col_b:
        prompt_b = st.text_area("Prompt B", height=150, key="pb",
                                placeholder="Version B of your prompt…")

    if st.button("Run A/B", type="primary",
                 disabled=not (prompt_a.strip() and prompt_b.strip())):
        res_a = _run_and_record(prompt_a.strip(), ab_model, judge_model, "ab-a," + tags_input)
        res_b = _run_and_record(prompt_b.strip(), ab_model, judge_model, "ab-b," + tags_input)

        col_a2, col_b2 = st.columns(2)

        for col, res, label in [(col_a2, res_a, "A"), (col_b2, res_b, "B")]:
            with col:
                st.subheader(f"Prompt {label}")
                st.metric("Quality", f"{res['quality_score']:.1f} / 10")
                st.metric("Speed", f"{res['tokens_per_second']:.1f} tok/s")
                st.metric("Latency", f"{res['total_duration_ms'] / 1000:.2f} s")
                st.metric("Out tokens", res["output_tokens"])
                st.plotly_chart(
                    _score_radar(res, f"Prompt {label}"),
                    use_container_width=True,
                )
                with st.expander("Response"):
                    st.markdown(res["response"])

        # Winner badge
        if res_a["quality_score"] > res_b["quality_score"]:
            st.success(f"Winner: **Prompt A** (quality {res_a['quality_score']:.1f} vs {res_b['quality_score']:.1f})")
        elif res_b["quality_score"] > res_a["quality_score"]:
            st.success(f"Winner: **Prompt B** (quality {res_b['quality_score']:.1f} vs {res_a['quality_score']:.1f})")
        else:
            st.info("Tie — both prompts scored equally.")


# ─────────────────────────────────────────────
# TAB 3 — DASHBOARD
# ─────────────────────────────────────────────
with tab_dash:
    st.header("Dashboard")

    runs = db.get_all_runs()
    if not runs:
        st.info("No runs yet.  Go to **Run** to test your first prompt.")
    else:
        df = pd.DataFrame(runs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["efficiency"] = df["quality_score"] / df["output_tokens"].replace(0, 1)

        # ── top KPIs ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total runs", len(df))
        k2.metric("Avg quality", f"{df['quality_score'].mean():.2f} / 10")
        k3.metric("Avg speed", f"{df['tokens_per_second'].mean():.1f} tok/s")
        k4.metric(
            "Hallucinations",
            f"{df['hallucination_detected'].sum()} "
            f"({df['hallucination_detected'].mean() * 100:.0f}%)",
        )

        st.divider()

        col1, col2 = st.columns(2)

        # Quality over time
        with col1:
            daily = (
                df.groupby("date")["quality_score"]
                .mean()
                .reset_index()
                .rename(columns={"quality_score": "avg_quality"})
            )
            fig_time = px.line(
                daily, x="date", y="avg_quality",
                title="Average quality score over time",
                markers=True,
                range_y=[0, 10],
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Quality per model (box plot)
        with col2:
            fig_box = px.box(
                df, x="model", y="quality_score",
                title="Quality distribution by model",
                color="model",
                range_y=[0, 10],
            )
            st.plotly_chart(fig_box, use_container_width=True)

        col3, col4 = st.columns(2)

        # Efficiency scatter: latency vs quality, bubble = output tokens
        with col3:
            fig_scatter = px.scatter(
                df,
                x="total_duration_ms",
                y="quality_score",
                size="output_tokens",
                color="model",
                hover_data=["prompt_text"],
                title="Latency vs quality (bubble = output tokens)",
                labels={"total_duration_ms": "Latency (ms)", "quality_score": "Quality"},
                range_y=[0, 10],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Tokens per second per model
        with col4:
            fig_tps = px.bar(
                df.groupby("model")["tokens_per_second"].mean().reset_index(),
                x="model", y="tokens_per_second",
                title="Average generation speed (tok/s)",
                color="model",
            )
            st.plotly_chart(fig_tps, use_container_width=True)

        # Sub-score breakdown
        st.subheader("Score dimensions by model")
        score_cols = ["relevance_score", "accuracy_score",
                      "completeness_score", "conciseness_score"]
        score_means = (
            df.groupby("model")[score_cols]
            .mean()
            .reset_index()
            .melt(id_vars="model", var_name="dimension", value_name="score")
        )
        score_means["dimension"] = (
            score_means["dimension"].str.replace("_score", "").str.capitalize()
        )
        fig_dim = px.bar(
            score_means, x="dimension", y="score", color="model",
            barmode="group",
            title="Average sub-scores by model",
            range_y=[0, 10],
        )
        st.plotly_chart(fig_dim, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 — HISTORY
# ─────────────────────────────────────────────
with tab_hist:
    st.header("Run history")

    runs = db.get_all_runs()
    if not runs:
        st.info("No runs recorded yet.")
    else:
        df_hist = pd.DataFrame(runs)

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            model_filter = st.multiselect(
                "Filter by model",
                df_hist["model"].unique().tolist(),
            )
        with fc2:
            min_q = st.slider("Minimum quality score", 0.0, 10.0, 0.0, 0.5)

        if model_filter:
            df_hist = df_hist[df_hist["model"].isin(model_filter)]
        df_hist = df_hist[df_hist["quality_score"] >= min_q]

        display_cols = [
            "id", "timestamp", "model", "prompt_text",
            "quality_score", "input_tokens", "output_tokens",
            "tokens_per_second", "total_duration_ms",
            "hallucination_detected", "judge_model", "tags",
        ]
        st.dataframe(
            df_hist[display_cols].rename(columns={
                "total_duration_ms": "latency_ms",
                "hallucination_detected": "hallucination",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Expand a single run
        st.subheader("Inspect response")
        run_ids = df_hist["id"].tolist()
        if run_ids:
            chosen_id = st.selectbox("Select run ID", run_ids)
            row = df_hist[df_hist["id"] == chosen_id].iloc[0]
            st.markdown(f"**Prompt:** {row['prompt_text']}")
            st.markdown(f"**Response:**\n\n{row['response_text']}")

            # Delete
            if st.button(f"Delete run #{chosen_id}", type="secondary"):
                db.delete_run(chosen_id)
                st.success(f"Run #{chosen_id} deleted.")
                st.rerun()

        # CSV download
        csv = df_hist[display_cols].to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="prompt_efficiency_runs.csv",
            mime="text/csv",
        )

"""
Norm-Constrained Dialogue Framework — Streamlit Dashboard

Sections:
  1. Overview
  2. Scenario Explorer
  3. Transcript Viewer
  4. Metrics Comparison
  5. Strategy Leaderboard
  6. Error Analysis

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from norm_dialogue_framework.config import load_config
from norm_dialogue_framework.visualization.plots import (
    plot_radar_chart,
    plot_score_heatmap,
    plot_strategy_comparison,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Norm-Constrained Dialogue Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        border-left: 4px solid #3498db;
    }
    .risk-high { border-left-color: #e74c3c; }
    .risk-medium { border-left-color: #f39c12; }
    .risk-low { border-left-color: #2ecc71; }
    .strategy-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_COLOURS = {
    "baseline": "#e74c3c",
    "rule_augmented": "#f39c12",
    "critique_revise": "#3498db",
    "candidate_selector": "#2ecc71",
    "constrained_filter": "#9b59b6",
}

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
SAMPLE_RUNS_DIR = RESULTS_DIR / "sample_runs"
SYNTHETIC_DIR = Path("data/synthetic")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data
def load_summary_df() -> pd.DataFrame:
    path = TABLES_DIR / "experiment_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback: load from sample_runs JSON files
    return _load_summaries_from_runs()


@st.cache_data
def load_scenarios_df() -> pd.DataFrame:
    path = SYNTHETIC_DIR / "scenarios.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_episodes() -> list[dict]:
    episodes = []
    for f in sorted(SAMPLE_RUNS_DIR.glob("episode_*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                episodes.append(json.load(fh))
        except Exception:
            pass
    return episodes


def _load_summaries_from_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(SAMPLE_RUNS_DIR.glob("episode_*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                ep = json.load(fh)
            turns = [t for t in ep.get("turns", []) if t.get("speaker") == "agent"]
            if not turns:
                continue
            metrics = [t.get("turn_metrics") for t in turns if t.get("turn_metrics")]
            if not metrics:
                continue

            def mean_m(key):
                vals = [m.get(key, 0) for m in metrics]
                return sum(vals) / len(vals) if vals else 0.0

            rows.append(
                {
                    "episode_id": ep.get("episode_id", ""),
                    "agent_strategy": ep.get("agent_strategy", "unknown"),
                    "scenario_type": ep.get("scenario", {}).get("scenario_type", ""),
                    "respondent_profile": ep.get("scenario", {}).get("respondent_profile", ""),
                    "sensitivity_level": ep.get("scenario", {}).get("sensitivity_level", ""),
                    "composite_score": mean_m("composite_score"),
                    "ethical_alignment_score": mean_m("ethical_alignment_score"),
                    "utility_score": mean_m("utility_score"),
                    "mean_coercion_risk": mean_m("coercion_risk"),
                    "mean_leading_question_risk": mean_m("leading_question_risk"),
                    "mean_empathy_score": mean_m("empathy_score"),
                    "mean_information_yield": mean_m("information_yield"),
                    "final_trust_level": ep.get("final_trust_level", 0.5),
                    "final_stress_level": ep.get("final_stress_level", 0.3),
                    "total_turns": ep.get("total_turns", 0),
                }
            )
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    st.sidebar.image(
        "https://via.placeholder.com/300x80/1a1a2e/ffffff?text=NCDF+Research",
        use_column_width=True,
    )
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Scenario Explorer",
            "Transcript Viewer",
            "Metrics Comparison",
            "Strategy Leaderboard",
            "Error Analysis",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Research Sandbox Only**\n\n"
        "All scenarios are synthetic and fictional. "
        "This dashboard visualises research simulation outputs only."
    )
    return section


# ---------------------------------------------------------------------------
# Section: Overview
# ---------------------------------------------------------------------------


def render_overview(df: pd.DataFrame) -> None:
    st.title("🔬 Norm-Constrained Dialogue Framework")
    st.markdown(
        """
        **Research simulator for evaluating AI alignment in ethically constrained conversations.**

        This dashboard visualises outputs from the NCDF pipeline, which generates synthetic
        dialogue scenarios, simulates conversations using multiple agent strategies, and
        evaluates each strategy on norm compliance and information utility.
        """
    )
    st.markdown("---")

    if df.empty:
        st.warning(
            "No experiment results found. Run `python scripts/run_experiments.py` first."
        )
        _render_quickstart()
        return

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Episodes", len(df))
    col2.metric("Strategies Tested", df["agent_strategy"].nunique() if "agent_strategy" in df.columns else 0)
    col3.metric(
        "Best Composite Score",
        f"{df['composite_score'].max():.3f}" if "composite_score" in df.columns else "N/A",
    )
    col4.metric(
        "Mean Ethical Score",
        f"{df['ethical_alignment_score'].mean():.3f}" if "ethical_alignment_score" in df.columns else "N/A",
    )
    col5.metric(
        "Mean Utility Score",
        f"{df['utility_score'].mean():.3f}" if "utility_score" in df.columns else "N/A",
    )

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Composite Score by Strategy")
        if "agent_strategy" in df.columns and "composite_score" in df.columns:
            fig = px.box(
                df,
                x="agent_strategy",
                y="composite_score",
                color="agent_strategy",
                color_discrete_map=STRATEGY_COLOURS,
                labels={"agent_strategy": "Strategy", "composite_score": "Composite Score"},
            )
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Episode Distribution by Profile")
        if "respondent_profile" in df.columns:
            fig2 = px.pie(
                df,
                names="respondent_profile",
                title="Respondent Profile Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig2, use_container_width=True)


def _render_quickstart() -> None:
    st.subheader("Quick Start")
    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic scenarios
python scripts/generate_dataset.py --n 50

# 3. Run simulation (single episode)
python scripts/run_simulation.py --strategy candidate_selector

# 4. Run full experiment
python scripts/run_experiments.py --n-episodes 10

# 5. Launch this dashboard
streamlit run app/dashboard.py
        """,
        language="bash",
    )


# ---------------------------------------------------------------------------
# Section: Scenario Explorer
# ---------------------------------------------------------------------------


def render_scenario_explorer() -> None:
    st.title("📋 Scenario Explorer")
    df = load_scenarios_df()

    if df.empty:
        st.warning("No scenario data found. Run `python scripts/generate_dataset.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        scenario_filter = st.selectbox(
            "Scenario Type", ["All"] + sorted(df["scenario_type"].unique().tolist())
        )
    with col2:
        profile_filter = st.selectbox(
            "Respondent Profile", ["All"] + sorted(df["respondent_profile"].unique().tolist())
        )
    with col3:
        sensitivity_filter = st.selectbox(
            "Sensitivity Level", ["All"] + sorted(df["sensitivity_level"].unique().tolist())
        )

    filtered = df.copy()
    if scenario_filter != "All":
        filtered = filtered[filtered["scenario_type"] == scenario_filter]
    if profile_filter != "All":
        filtered = filtered[filtered["respondent_profile"] == profile_filter]
    if sensitivity_filter != "All":
        filtered = filtered[filtered["sensitivity_level"] == sensitivity_filter]

    st.markdown(f"**{len(filtered)} scenarios** match the current filter.")

    if not filtered.empty:
        display_cols = [c for c in ["case_id", "scenario_type", "respondent_profile", "sensitivity_level", "context_summary"] if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(50), use_container_width=True)

        # Distribution plots
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                filtered["scenario_type"].value_counts().reset_index(),
                x="scenario_type",
                y="count",
                title="Scenario Types",
                color="scenario_type",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.bar(
                filtered["respondent_profile"].value_counts().reset_index(),
                x="respondent_profile",
                y="count",
                title="Respondent Profiles",
                color="respondent_profile",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Section: Transcript Viewer
# ---------------------------------------------------------------------------


def render_transcript_viewer() -> None:
    st.title("💬 Transcript Viewer")
    episodes = load_episodes()

    if not episodes:
        st.warning(
            "No episode transcripts found. Run `python scripts/run_simulation.py` first."
        )
        return

    ep_options = {
        f"{ep.get('episode_id', '?')} | {ep.get('agent_strategy', '?')} | {ep.get('scenario', {}).get('respondent_profile', '?')}": ep
        for ep in episodes
    }
    selected_key = st.selectbox("Select Episode", list(ep_options.keys()))
    ep = ep_options[selected_key]

    # Metadata
    scenario = ep.get("scenario", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy", ep.get("agent_strategy", "N/A"))
    col2.metric("Profile", scenario.get("respondent_profile", "N/A"))
    col3.metric("Scenario Type", scenario.get("scenario_type", "N/A"))
    col4.metric("Sensitivity", scenario.get("sensitivity_level", "N/A"))

    with st.expander("Context"):
        st.write(scenario.get("context_summary", "N/A"))

    st.markdown("---")
    st.subheader("Conversation Transcript")

    for turn in ep.get("turns", []):
        speaker = turn.get("speaker", "?").upper()
        utterance = turn.get("utterance", "")
        metrics = turn.get("turn_metrics")

        if speaker == "AGENT":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(utterance)
                if metrics:
                    cols = st.columns(4)
                    cols[0].caption(f"Composite: {metrics.get('composite_score', 0):.2f}")
                    cols[1].caption(f"Ethical: {metrics.get('ethical_alignment_score', 0):.2f}")
                    cols[2].caption(f"Coercion risk: {metrics.get('coercion_risk', 0):.2f}")
                    cols[3].caption(f"Empathy: {metrics.get('empathy_score', 0):.2f}")
        else:
            with st.chat_message("user", avatar="👤"):
                st.write(utterance)
                trust = turn.get("respondent_trust_level")
                stress = turn.get("respondent_stress_level")
                if trust is not None:
                    st.caption(f"Trust: {trust:.2f} | Stress: {stress:.2f}")


# ---------------------------------------------------------------------------
# Section: Metrics Comparison
# ---------------------------------------------------------------------------


def render_metrics_comparison(df: pd.DataFrame) -> None:
    st.title("📊 Metrics Comparison")

    if df.empty:
        st.warning("No data available. Run `python scripts/run_experiments.py` first.")
        return

    metric_options = [c for c in df.columns if c not in {"episode_id", "case_id", "scenario_type", "respondent_profile", "sensitivity_level", "agent_strategy", "completed"}]

    selected_metric = st.selectbox("Select Metric", metric_options, index=metric_options.index("composite_score") if "composite_score" in metric_options else 0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution by Strategy")
        fig = px.violin(
            df,
            x="agent_strategy",
            y=selected_metric,
            color="agent_strategy",
            box=True,
            points="all",
            color_discrete_map=STRATEGY_COLOURS,
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution by Respondent Profile")
        if "respondent_profile" in df.columns:
            fig2 = px.box(
                df,
                x="respondent_profile",
                y=selected_metric,
                color="respondent_profile",
            )
            fig2.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Heatmap: Strategy × Metric")
    heatmap_metrics = [
        m for m in [
            "composite_score", "ethical_alignment_score", "utility_score",
            "mean_coercion_risk", "mean_empathy_score", "mean_information_yield",
            "mean_neutrality_score", "final_trust_level",
        ] if m in df.columns
    ]
    pivot = df.groupby("agent_strategy")[heatmap_metrics].mean().reset_index()
    pivot_melted = pivot.melt(id_vars="agent_strategy", value_vars=heatmap_metrics, var_name="metric", value_name="value")
    fig3 = px.density_heatmap(
        pivot_melted,
        x="metric",
        y="agent_strategy",
        z="value",
        color_continuous_scale="RdYlGn",
        title="Mean Score Heatmap",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Radar chart
    st.subheader("Radar Chart")
    try:
        radar_fig = plot_radar_chart(df)
        st.plotly_chart(radar_fig, use_container_width=True)
    except Exception as e:
        st.info(f"Radar chart unavailable: {e}")


# ---------------------------------------------------------------------------
# Section: Strategy Leaderboard
# ---------------------------------------------------------------------------


def render_leaderboard(df: pd.DataFrame) -> None:
    st.title("🏆 Strategy Leaderboard")

    if df.empty:
        st.warning("No data available. Run `python scripts/run_experiments.py` first.")
        return

    agg_cols = [c for c in ["composite_score", "ethical_alignment_score", "utility_score", "mean_coercion_risk", "mean_empathy_score", "mean_information_yield", "final_trust_level"] if c in df.columns]
    leaderboard = (
        df.groupby("agent_strategy")[agg_cols]
        .mean()
        .round(3)
        .sort_values("composite_score", ascending=False)
        .reset_index()
    )

    # Rank
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

    st.dataframe(leaderboard.style.background_gradient(subset=agg_cols, cmap="RdYlGn"), use_container_width=True)

    # Bar chart
    st.subheader("Composite Score Ranking")
    fig = px.bar(
        leaderboard,
        x="agent_strategy",
        y="composite_score",
        color="agent_strategy",
        color_discrete_map=STRATEGY_COLOURS,
        text="composite_score",
        title="Mean Composite Score by Strategy",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1.1], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Win rate table
    st.subheader("Per-Profile Performance")
    if "respondent_profile" in df.columns:
        pivot = df.pivot_table(
            values="composite_score",
            index="agent_strategy",
            columns="respondent_profile",
            aggfunc="mean",
        ).round(3)
        st.dataframe(pivot.style.background_gradient(cmap="RdYlGn"), use_container_width=True)


# ---------------------------------------------------------------------------
# Section: Error Analysis
# ---------------------------------------------------------------------------


def render_error_analysis(df: pd.DataFrame) -> None:
    st.title("⚠️ Error Analysis")

    if df.empty:
        st.warning("No data available. Run `python scripts/run_experiments.py` first.")
        return

    st.markdown(
        """
        Identifies episodes with high norm violation risk or low performance,
        useful for diagnosing failure modes in alignment strategies.
        """
    )

    threshold = st.slider("Coercion Risk Threshold", 0.0, 1.0, 0.3, 0.05)

    risk_cols = [c for c in ["mean_coercion_risk", "mean_leading_question_risk", "mean_pressure_escalation_risk"] if c in df.columns]

    if risk_cols:
        high_risk = df[df[risk_cols].max(axis=1) >= threshold]
        st.markdown(f"**{len(high_risk)} episodes** exceed the threshold on at least one risk metric.")

        if not high_risk.empty:
            display_cols = [c for c in ["episode_id", "agent_strategy", "respondent_profile", "scenario_type"] + risk_cols + ["composite_score"] if c in high_risk.columns]
            st.dataframe(high_risk[display_cols].sort_values(risk_cols[0], ascending=False).head(20), use_container_width=True)

    # Strategy failure rate
    st.subheader("Low-Composite Episodes per Strategy")
    if "composite_score" in df.columns and "agent_strategy" in df.columns:
        low_thresh = st.slider("Low Composite Score Threshold", 0.0, 0.8, 0.4, 0.05)
        low_df = df[df["composite_score"] < low_thresh]
        failure_rate = low_df.groupby("agent_strategy").size() / df.groupby("agent_strategy").size()
        failure_df = failure_rate.reset_index()
        failure_df.columns = ["agent_strategy", "failure_rate"]
        fig = px.bar(
            failure_df,
            x="agent_strategy",
            y="failure_rate",
            color="agent_strategy",
            color_discrete_map=STRATEGY_COLOURS,
            title=f"Fraction of Episodes with Composite Score < {low_thresh:.2f}",
            labels={"failure_rate": "Failure Rate"},
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.subheader("Metric Correlations")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        corr = df[numeric_cols].corr()
        fig2 = px.imshow(corr, color_continuous_scale="RdBu_r", title="Metric Correlation Matrix", text_auto=".2f")
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    section = render_sidebar()
    df = load_summary_df()

    if section == "Overview":
        render_overview(df)
    elif section == "Scenario Explorer":
        render_scenario_explorer()
    elif section == "Transcript Viewer":
        render_transcript_viewer()
    elif section == "Metrics Comparison":
        render_metrics_comparison(df)
    elif section == "Strategy Leaderboard":
        render_leaderboard(df)
    elif section == "Error Analysis":
        render_error_analysis(df)

    # Footer
    st.markdown("---")
    st.caption(
        "**Norm-Constrained Dialogue Framework** · Research simulator · All scenarios synthetic · "
        "Not a production or operational system."
    )


if __name__ == "__main__":
    main()

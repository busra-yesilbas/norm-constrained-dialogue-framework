"""
Visualisation utilities for the Norm-Constrained Dialogue Framework.

Provides plotting functions for:
- Strategy comparison bar charts
- Metric distributions (violin / box plots)
- Score heatmaps
- Trust/stress trajectory plots
- Radar (spider) charts for multi-metric comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use("Agg")  # Non-interactive backend for script/server use

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_STRATEGY_COLOURS = {
    "baseline": "#e74c3c",
    "rule_augmented": "#f39c12",
    "critique_revise": "#3498db",
    "candidate_selector": "#2ecc71",
    "constrained_filter": "#9b59b6",
}

_DEFAULT_COLOUR = "#95a5a6"


def _strategy_colour(strategy: str) -> str:
    return _STRATEGY_COLOURS.get(strategy, _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------


def plot_strategy_comparison(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Bar chart comparing strategies on aggregate scores.

    Parameters
    ----------
    df:
        DataFrame with columns including ``agent_strategy`` and metric columns.
    metrics:
        Metric columns to plot.  Defaults to composite, ethical, and utility scores.
    output_path:
        If provided, save the figure to this path.
    use_plotly:
        Use Plotly (interactive) instead of Matplotlib.
    """
    if metrics is None:
        metrics = ["composite_score", "ethical_alignment_score", "utility_score"]

    available = [m for m in metrics if m in df.columns]
    summary = df.groupby("agent_strategy")[available].mean().reset_index()

    if use_plotly:
        melted = summary.melt(id_vars="agent_strategy", value_vars=available, var_name="metric", value_name="score")
        fig = px.bar(
            melted,
            x="agent_strategy",
            y="score",
            color="metric",
            barmode="group",
            title="Strategy Comparison — Aggregate Scores",
            labels={"agent_strategy": "Strategy", "score": "Mean Score"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 1])
        if output_path:
            fig.write_html(str(output_path))
        return fig

    # Matplotlib
    strategies = summary["agent_strategy"].tolist()
    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(available):
        values = summary[metric].tolist()
        bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Score")
    ax.set_title("Strategy Comparison — Aggregate Scores")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Metric distributions
# ---------------------------------------------------------------------------


def plot_metric_distributions(
    df: pd.DataFrame,
    metric: str = "composite_score",
    output_path: Optional[str | Path] = None,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Box/violin plot of a metric distribution per strategy."""
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in DataFrame.")

    if use_plotly:
        fig = px.violin(
            df,
            x="agent_strategy",
            y=metric,
            box=True,
            points="all",
            color="agent_strategy",
            title=f"Distribution of {metric.replace('_', ' ').title()} by Strategy",
            color_discrete_map=_STRATEGY_COLOURS,
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        if output_path:
            fig.write_html(str(output_path))
        return fig

    strategies = df["agent_strategy"].unique()
    data = [df[df["agent_strategy"] == s][metric].dropna().tolist() for s in strategies]
    colours = [_strategy_colour(s) for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, (pc, colour) in enumerate(zip(parts["bodies"], colours)):
        pc.set_facecolor(colour)
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def plot_score_heatmap(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Heatmap of mean scores per strategy × metric."""
    if metrics is None:
        metrics = [
            "ethical_alignment_score",
            "utility_score",
            "composite_score",
            "mean_coercion_risk",
            "mean_leading_question_risk",
            "mean_empathy_score",
            "mean_information_yield",
        ]
    available = [m for m in metrics if m in df.columns]
    pivot = df.groupby("agent_strategy")[available].mean()

    if use_plotly:
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Mean Scores Heatmap: Strategy × Metric",
        )
        if output_path:
            fig.write_html(str(output_path))
        return fig

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([m.replace("_", "\n") for m in available], fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Mean Scores Heatmap: Strategy × Metric")

    for i in range(len(pivot.index)):
        for j in range(len(available)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Trust / stress trajectories
# ---------------------------------------------------------------------------


def plot_trust_stress_trajectories(
    episode_data: list[dict],
    output_path: Optional[str | Path] = None,
    use_plotly: bool = False,
) -> go.Figure | plt.Figure:
    """Line plot of trust and stress across dialogue turns for one episode.

    Parameters
    ----------
    episode_data:
        List of dicts with keys: 'turn_id', 'speaker', 'trust', 'stress'.
    """
    turns = [d for d in episode_data if d.get("speaker") == "respondent"]
    if not turns:
        turns = episode_data

    x = list(range(len(turns)))
    trusts = [d.get("trust", 0.5) for d in turns]
    stresses = [d.get("stress", 0.3) for d in turns]

    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=trusts, mode="lines+markers", name="Trust", line=dict(color="#2ecc71")))
        fig.add_trace(go.Scatter(x=x, y=stresses, mode="lines+markers", name="Stress", line=dict(color="#e74c3c")))
        fig.update_layout(
            title="Respondent Trust & Stress Across Turns",
            xaxis_title="Turn",
            yaxis_title="Level",
            yaxis_range=[0, 1],
        )
        if output_path:
            fig.write_html(str(output_path))
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, trusts, "g-o", label="Trust", linewidth=2)
    ax.plot(x, stresses, "r-s", label="Stress", linewidth=2)
    ax.set_xlabel("Respondent Turn")
    ax.set_ylabel("Level")
    ax.set_ylim(0, 1)
    ax.set_title("Respondent Trust & Stress Across Turns")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------


def plot_radar_chart(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Plotly radar chart comparing strategies on multiple metrics."""
    if metrics is None:
        metrics = [
            "ethical_alignment_score",
            "utility_score",
            "mean_empathy_score",
            "mean_neutrality_score",
            "mean_information_yield",
            "mean_trust_preservation",
        ]
    available = [m for m in metrics if m in df.columns]
    summary = df.groupby("agent_strategy")[available].mean()

    fig = go.Figure()
    for strategy, row in summary.iterrows():
        values = row[available].tolist()
        values += [values[0]]  # close the polygon
        labels = [m.replace("mean_", "").replace("_score", "").replace("_", " ").title() for m in available]
        labels += [labels[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=str(strategy),
                line=dict(color=_strategy_colour(str(strategy))),
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Multi-Metric Strategy Comparison (Radar)",
        showlegend=True,
    )

    if output_path:
        fig.write_html(str(output_path))
    return fig

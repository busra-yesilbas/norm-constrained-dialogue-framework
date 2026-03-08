"""
Configuration management for the Norm-Constrained Dialogue Framework.

Loads YAML configuration files and provides a typed, validated config
object. Supports environment variable overrides via python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    n_scenarios: int = 100
    output_dir: str = "data/synthetic"
    scenario_types: list[str] = Field(default_factory=list)
    respondent_profiles: list[str] = Field(default_factory=list)
    sensitivity_levels: list[str] = Field(default_factory=list)


class SimulationConfig(BaseModel):
    max_turns: int = 12
    min_turns: int = 4
    trust_initial_range: tuple[float, float] = (0.3, 0.8)
    stress_initial_range: tuple[float, float] = (0.1, 0.7)
    temperature: float = 0.7


class CandidateSelectorConfig(BaseModel):
    n_candidates: int = 5


class CritiqueReviseConfig(BaseModel):
    max_revisions: int = 2


class AgentsConfig(BaseModel):
    strategies: list[str] = Field(default_factory=list)
    candidate_selector: CandidateSelectorConfig = Field(
        default_factory=CandidateSelectorConfig
    )
    critique_revise: CritiqueReviseConfig = Field(default_factory=CritiqueReviseConfig)
    llm_backend: Optional[str] = None


class EvaluationConfig(BaseModel):
    metrics_config: str = "configs/metrics.yaml"
    save_transcripts: bool = True
    output_dir: str = "results"


class ExperimentsConfig(BaseModel):
    n_episodes_per_strategy: int = 20
    parallel_workers: int = 1
    save_all_transcripts: bool = False
    summary_output: str = "results/tables/experiment_summary.csv"


class ProjectConfig(BaseModel):
    name: str = "Norm-Constrained Dialogue Framework"
    version: str = "0.1.0"
    random_seed: int = 42
    log_level: str = "INFO"


class FrameworkConfig(BaseModel):
    """Top-level validated configuration object."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path = "configs/default.yaml") -> FrameworkConfig:
    """Load and validate the framework configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  Relative paths are resolved from the
        current working directory (i.e. the project root).

    Returns
    -------
    FrameworkConfig
        A fully validated configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        # Return defaults if config file is missing (useful for tests)
        return FrameworkConfig()

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply environment variable overrides
    if seed := os.getenv("RANDOM_SEED"):
        raw.setdefault("project", {})["random_seed"] = int(seed)
    if log_level := os.getenv("LOG_LEVEL"):
        raw.setdefault("project", {})["log_level"] = log_level
    if results_dir := os.getenv("RESULTS_DIR"):
        raw.setdefault("evaluation", {})["output_dir"] = results_dir
    if llm_backend := os.getenv("LLM_BACKEND"):
        raw.setdefault("agents", {})["llm_backend"] = llm_backend or None

    return FrameworkConfig(**raw)


def load_metrics_config(config_path: str | Path = "configs/metrics.yaml") -> dict[str, Any]:
    """Load the metrics configuration YAML and return as a plain dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

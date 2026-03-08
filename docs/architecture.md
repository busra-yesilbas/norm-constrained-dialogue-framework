# System Architecture

## Overview

The Norm-Constrained Dialogue Framework (NCDF) is a modular Python research
simulator.  It generates synthetic conversation scenarios, simulates ethically
constrained dialogues using multiple agent strategies, evaluates norm compliance
and information utility, and compares alignment approaches.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Norm-Constrained Dialogue Framework              │
├─────────────────────────────────────────────────────────────────────┤
│  configs/        ← YAML configuration (scenarios, metrics, defaults)│
│  data/synthetic/ ← Generated scenario datasets                      │
│  results/        ← Transcripts, evaluation tables, figures          │
└─────────────────────────────────────────────────────────────────────┘
                              │
               ┌──────────────┼──────────────────────┐
               ▼              ▼                       ▼
    ┌─────────────────┐  ┌─────────────┐  ┌──────────────────────┐
    │  Data Module    │  │  Agents     │  │  Evaluation          │
    │─────────────────│  │─────────────│  │──────────────────────│
    │ SyntheticGen    │  │ BaseAgent   │  │ RuleChecker          │
    │ Scenario schema │  │ Baseline    │  │ MetricsComputer      │
    └────────┬────────┘  │ RuleAug     │  │ RewardModel          │
             │           │ CritiqueRev │  │ Evaluator            │
             ▼           │ CandidateSel│  └──────────┬───────────┘
    ┌─────────────────┐  │ Constrained │             │
    │  Simulation     │  └──────┬──────┘             │
    │─────────────────│         │                    │
    │ RespondentSim   │◄────────┘                    │
    │ ProfileConfigs  │                              │
    │ DialogueRunner  │──────────────────────────────►
    └─────────────────┘    DialogueEpisode            │
                                                      ▼
                                          ┌──────────────────────┐
                                          │  Experiments         │
                                          │──────────────────────│
                                          │ StrategyComparison   │
                                          │ ExperimentResult     │
                                          └──────────┬───────────┘
                                                     │
                              ┌──────────────────────┼──────────────────┐
                              ▼                       ▼                  ▼
                   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐
                   │  Visualization   │  │  Streamlit App  │  │  Notebooks   │
                   │─────────────────-│  │─────────────────│  │──────────────│
                   │ Matplotlib       │  │ Dashboard       │  │ 01-04        │
                   │ Plotly           │  │ 6 sections      │  │ Exploratory  │
                   └──────────────────┘  └─────────────────┘  └──────────────┘
```

---

## Module Descriptions

### `norm_dialogue_framework.config`
Loads and validates YAML configuration files using Pydantic models.
Supports environment variable overrides (`.env`).

**Key class:** `FrameworkConfig` — validated top-level config object.

---

### `norm_dialogue_framework.schemas`
Pydantic data models for all domain objects.

| Class | Purpose |
|---|---|
| `Scenario` | A fully-specified synthetic conversation scenario |
| `DialogueTurn` | A single turn (agent or respondent) with metrics |
| `DialogueEpisode` | Complete conversation with all turns and metadata |
| `TurnMetrics` | Per-turn evaluation scores |
| `EpisodeSummary` | Aggregated episode-level summary for reporting |
| `ExperimentResult` | Container for all summaries in an experiment run |

---

### `norm_dialogue_framework.data`
Synthetic scenario generation.

**`SyntheticScenarioGenerator`**
- Generates fictional scenarios from template libraries
- Supports filtering by scenario type and respondent profile
- Outputs JSON + CSV for downstream use
- Fully reproducible with seed

---

### `norm_dialogue_framework.agents`

All agents implement the interface:
```python
def generate_next_turn(self, state: DialogueState) -> str: ...
```

| Agent | Alignment Strategy |
|---|---|
| `BaselineAgent` | No norm constraints; direct template selection |
| `RuleAugmentedAgent` | Scores candidates against rule-based norm checks |
| `CritiqueReviseAgent` | Generates draft, critiques, revises up to N times |
| `CandidateSelectorAgent` | Best-of-N sampling with composite reward function |
| `ConstrainedFilterAgent` | Hard-rejects norm-violating responses; uses safe fallbacks |

**Fallback mode:** All agents run in template-based mode when no LLM backend
is configured (`LLM_BACKEND=` in `.env`).

**LLM mode:** When `LLM_BACKEND=openai` and a valid API key are set, agents
delegate to an OpenAI-compatible endpoint with norm-aware prompts.

---

### `norm_dialogue_framework.simulation`

**`RespondentSimulator`**
- Maintains internal trust, stress, and fatigue state
- Updates state based on coercion / empathy signals in agent utterances
- Produces contextually appropriate responses from template libraries
- Behaviour modulated by behavioural profile config

**`DialogueRunner`**
- Orchestrates turn-by-turn dialogue
- Calls agent for utterance generation
- Calls respondent for response
- Tracks goal progress, detects early stopping conditions
- Optionally evaluates each agent turn in real time

**`RespondentProfileConfig` / `PROFILE_CONFIGS`**
- Defines parameters per respondent profile (trust range, coercion
  sensitivity, hesitation probability, etc.)

---

### `norm_dialogue_framework.evaluation`

**`RuleChecker`**
- Pattern-based checks for coercion, leading language, empathy, transparency,
  neutrality, and procedural fairness
- Returns a `RuleCheckResult` with per-check scores

**`MetricsComputer`**
- Combines rule check results with utility heuristics
- Produces a `TurnMetrics` object with all per-turn scores
- Computes weighted aggregate: ethical, utility, composite

**`RewardModel`**
- Scores candidate utterances with a composite reward signal
- Adds hard penalties for critical violations
- Used by `CandidateSelectorAgent` for Best-of-N selection

**`Evaluator`**
- Online: evaluate each agent turn in real time during simulation
- Offline: post-hoc evaluation of saved `DialogueEpisode` objects
- Produces `EpisodeSummary` with aggregated metrics

---

### `norm_dialogue_framework.experiments`

**`StrategyComparison`**
- Runs N episodes per strategy across all configured strategies
- Uses `tqdm` progress bars
- Saves CSV and JSON outputs
- Returns `ExperimentResult` for downstream analysis

---

### `norm_dialogue_framework.visualization`

| Function | Chart Type |
|---|---|
| `plot_strategy_comparison` | Grouped bar (Matplotlib or Plotly) |
| `plot_metric_distributions` | Violin plot |
| `plot_score_heatmap` | Heatmap |
| `plot_trust_stress_trajectories` | Line plot |
| `plot_radar_chart` | Radar/spider chart (Plotly) |

---

## Data Flow

```
1. SyntheticScenarioGenerator.generate()
   → list[Scenario]

2. DialogueRunner.run(scenario, agent)
   → DialogueEpisode (with per-turn TurnMetrics via Evaluator)

3. Evaluator.evaluate_episode(episode)
   → EpisodeSummary

4. StrategyComparison.run()
   → ExperimentResult (list[EpisodeSummary])

5. plots.*(df)  /  dashboard.py
   → Figures, tables, interactive charts
```

---

## LLM Backend Integration

The framework supports an optional LLM backend:

```
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

When set, agents construct norm-aware prompts and delegate generation
to the LLM.  The system falls back gracefully to template mode if the
API call fails.

**No LLM is required** to run any part of the framework.  All
experiments, evaluations, and the dashboard work fully in fallback mode.

---

## Extensibility

To add a new agent strategy:
1. Create a class in `src/norm_dialogue_framework/agents/` extending `BaseAgent`
2. Implement `generate_next_turn(self, state: DialogueState) -> str`
3. Register in `agents/__init__.py` and `experiments/compare_strategies.py`

To add a new scenario type:
1. Add templates to `data/synthetic_generator.py`
2. Register in `configs/scenarios.yaml`
3. Add to `ScenarioType` enum in `schemas.py`

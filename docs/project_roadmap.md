# Project Roadmap

## Current Status: v0.1.0 — Research Prototype

The initial release provides a fully functional research simulator with:
- Synthetic scenario generation (5 types, 6 profiles)
- 5 alignment strategy implementations
- Rule-based evaluation pipeline
- Streamlit dashboard
- Reproducible experiment runner
- 4 analysis notebooks

---

## Phase 1: Core Research Simulator (✅ Complete — v0.1.0)

- [x] Modular Python package (`norm_dialogue_framework`)
- [x] Synthetic scenario generator
- [x] 5 dialogue agent strategies
- [x] Respondent behavioural simulator
- [x] Dialogue runner with online evaluation
- [x] Rule-based norm checking (`RuleChecker`)
- [x] Metrics computation (`MetricsComputer`)
- [x] Heuristic reward model (`RewardModel`)
- [x] Full evaluation pipeline (`Evaluator`)
- [x] Multi-strategy comparison experiment (`StrategyComparison`)
- [x] Streamlit dashboard (6 sections)
- [x] 4 Jupyter notebooks
- [x] pytest test suite
- [x] Full documentation

---

## Phase 2: Richer Modelling (Planned — v0.2.0)

**Goal:** Improve the realism of behavioural simulation and evaluation.

- [ ] **Memory model for respondent:** Track specific claims made across turns
  to enable consistency checking
- [ ] **Goal-completion tracking:** Heuristic detection of when each communication
  goal has been adequately addressed
- [ ] **Multi-turn pressure escalation model:** More sophisticated escalation
  detection using sliding window analysis
- [ ] **Scenario diversity expansion:** Add 5+ new scenario types and additional
  profile variants (e.g., defensive, compliant-but-avoidant)
- [ ] **Latent truth evaluation:** Score how much of the latent truth state was
  elicited, given what the respondent was willing to share
- [ ] **Discourse coherence metric:** Basic lexical coherence scoring across turns
- [ ] **Transcript quality rating:** Automated quality score based on multiple
  dimensions

---

## Phase 3: LLM Integration (Planned — v0.3.0)

**Goal:** Enable seamless integration with open-source and commercial LLMs.

- [ ] **HuggingFace Transformers backend:** Support for local models
  (e.g., Mistral, Llama-3) as agent backends
- [ ] **Sentence-transformer semantic scoring:** Replace pattern-matching
  norm checks with embedding-based similarity scoring
- [ ] **LLM-as-judge evaluation:** Use a secondary LLM to evaluate norm
  compliance as an alternative to rule-based checks
- [ ] **Structured prompting library:** Curated, tested norm-aware prompt
  templates per scenario type and strategy
- [ ] **Async/parallel episode execution:** Accelerate multi-strategy
  experiments using `asyncio` or `concurrent.futures`

---

## Phase 4: Statistical Analysis (Planned — v0.4.0)

**Goal:** Rigorous statistical comparison of strategies.

- [ ] **Bootstrap confidence intervals:** Report uncertainty in all metric
  comparisons
- [ ] **Effect size computation:** Cohen's d / rank-biserial correlation for
  strategy pairwise comparisons
- [ ] **Bayesian scoring model:** Replace heuristic weights with a simple
  Bayesian model of norm compliance
- [ ] **Factor analysis of metrics:** Identify latent structure in the metric
  space (e.g., is empathy_score independent of neutrality_score?)
- [ ] **Sensitivity analysis:** Vary metric weights and report robustness of
  strategy rankings

---

## Phase 5: Open Science & Community (Planned — v0.5.0)

**Goal:** Enable external validation and contribution.

- [ ] **Public scenario dataset:** Release a standardised synthetic scenario
  dataset for community benchmarking
- [ ] **Annotation interface:** Streamlit-based tool for human annotation of
  agent utterances (ethical quality ratings)
- [ ] **Human-model comparison:** Compare automated metric scores against
  human annotator judgements to validate heuristics
- [ ] **Plugin architecture:** Formal interface for adding new agent strategies,
  scenario types, and evaluation metrics without modifying core code
- [ ] **CI/CD pipeline:** GitHub Actions for automated testing, linting, and
  notebook execution on push
- [ ] **Docker image:** Containerised deployment of the full framework and dashboard

---

## Known Limitations (All Versions)

These limitations are by design in the research framing:

1. All behavioural simulation is heuristic; not derived from empirical behavioural data
2. Norm metrics are lexical pattern-based, not semantic
3. Composite score weights are manually specified, not learned
4. No real-world validation against actual conversation outcomes
5. Template-based generation limits linguistic diversity without LLM backends
6. The framework does not model institutional, cultural, or legal context

---

## Contribution Notes

If extending this project:

1. Preserve the research-sandbox framing at all times
2. Do not introduce references to real institutions, case files, or individuals
3. Maintain comprehensive docstrings and type hints
4. Add tests for all new modules
5. Update `docs/methodology.md` for any new assumptions introduced
6. Follow the `pyproject.toml` linting and formatting standards

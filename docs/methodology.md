# Methodology

## Research Context

The Norm-Constrained Dialogue Framework (NCDF) is a research sandbox for studying
**AI alignment in ethically sensitive conversational settings**.  It is designed to:

- Simulate dialogue agents subject to explicit normative constraints
- Model diverse respondent behaviours computationally
- Evaluate and compare alignment strategies along multiple ethical and utility dimensions
- Support reproducible, transparent, and open research

This framework is strictly an academic research tool.  All scenarios are synthetic
and fictional.  The system does not represent a production deployment, operational
decision-support tool, or clinical instrument.

---

## 1. Synthetic Scenario Generation

### Design Rationale

Synthetic scenarios are used for three reasons:

1. **Ethical safety:** No real individuals, personal data, or case files are involved.
2. **Controlled variation:** Scenario properties can be precisely specified and balanced.
3. **Reproducibility:** Fixed seeds produce identical datasets across runs.

### Generation Approach

Scenarios are assembled from structured template libraries covering:
- Contextual framing (who, what, where)
- Communication goals aligned to the scenario type
- Risk flags indicating sensitivity dimensions
- Recommended normative constraints
- A latent truth state (ground truth visible to the simulator but not the agent)

Scenario types include: workplace incident review, public service complaint,
clinical-style intake, compliance interview, and witness-style recall.

### Limitations

- Templates are manually authored and may not capture the full diversity of
  real conversational contexts.
- Latent truth states are simplified structures; real situations involve far
  more contextual nuance.
- Scenario generation does not model institutional dynamics, power asymmetries,
  or legal constraints beyond high-level flags.

---

## 2. Respondent Behavioural Simulation

### Modelling Assumptions

The respondent simulator models conversational behaviour as a function of:

1. **Behavioural profile:** A fixed set of parameters defining baseline
   informativeness, trust responsiveness, hesitation probability, etc.
2. **Trust level:** Evolves dynamically based on coercive or empathetic signals.
3. **Stress level:** Increases with coercive turns and fatigue.
4. **Fatigue:** Accumulates monotonically across turns.

### Profile Types

Six profiles are implemented:

| Profile | Key Characteristics |
|---|---|
| `cooperative` | High initial trust, high informativeness, low hesitation |
| `anxious` | Moderate trust, high stress, high hesitation |
| `resistant` | Low trust, low informativeness, high coercion sensitivity |
| `confused` | Moderate trust, low consistency, high clarification needs |
| `fatigued` | Moderate trust, declining informativeness, high fatigue response |
| `trauma_sensitive` | Low trust, very high coercion sensitivity, high empathy responsiveness |

### Dynamic State Updates

At each turn:
- **Coercive signals** (e.g., "you must", "admit") reduce trust and increase stress
  by amounts proportional to `coercion_sensitivity`.
- **Empathetic signals** (e.g., "take your time", "I understand") improve trust and
  reduce stress by amounts proportional to `empathy_responsiveness`.
- **Fatigue** accumulates at a fixed rate, progressively reducing informativeness
  and increasing hesitation.

### Limitations

- The behavioural model is a computational abstraction, not a psychological model.
- It does not incorporate memory, narrative coherence, or theory of mind.
- Profile parameters are manually set, not learned from behavioural data.
- The simulation does not model physiological, cultural, or trauma-specific responses.

---

## 3. Norm-Constrained Generation

### Norm Framework

Seven primary norms are operationalised:

| Norm | Definition |
|---|---|
| `non_coercion` | Absence of pressure, threat, or demands |
| `non_leading` | Avoidance of presuppositive or suggestive language |
| `empathy` | Acknowledgement of respondent's emotional state |
| `transparency` | Clarity about the purpose of questions |
| `neutrality` | Absence of judgmental or biased language |
| `procedural_fairness` | Respect for respondent's rights and process integrity |
| `emotional_sensitivity` | Care for respondent's wellbeing across the conversation |

### Agent Strategies

Five alignment strategies are implemented and compared:

**1. Baseline** (no constraints)
- Selects questions from a template pool without any norm checking.
- Serves as the experimental control condition.

**2. Rule-Augmented**
- Generates a candidate pool and scores each against pattern-based norm rules.
- Selects the highest-scoring candidate.
- Represents explicit-rule augmentation of generation.

**3. Critique-and-Revise**
- Generates an initial draft, critiques it for norm violations, and applies
  heuristic rewrites iteratively.
- Inspired by Constitutional AI and self-critique prompting paradigms.

**4. Candidate Selector (Best-of-N)**
- Generates N candidate utterances and ranks them by composite reward.
- Reward = `ethical_weight × norm_score + utility_weight × utility_score`.
- Represents an offline reward optimisation / rejection sampling approach.

**5. Constrained Filter**
- Rejects any generated utterance containing a norm violation.
- Falls back to a curated safe-response library.
- Most conservative strategy; prioritises ethical safety over utility.

### Important Disclaimer

These strategies are **heuristic approximations** implemented for research
demonstration.  They are not:
- Trained neural reward models
- Production RLHF pipelines
- Validated clinical or legal assessment tools

They simulate, at a conceptual level, alignment paradigms described in the
research literature (e.g., Constitutional AI, RLHF, Best-of-N sampling).

---

## 4. Evaluation Design

### Per-Turn Metrics

Each agent turn is scored on:

**Ethical compliance metrics** (0 = worst, 1 = best after inversion):
- `coercion_risk`: Pattern-matched coercive language severity
- `leading_question_risk`: Pattern-matched suggestive language
- `pressure_escalation_risk`: Trend in coercion across recent turns
- `empathy_score`: Presence of empathetic language patterns
- `transparency_score`: Presence of purpose-clarifying language
- `neutrality_score`: Absence of judgmental language
- `procedural_fairness_score`: Presence of rights-affirming language

**Utility metrics**:
- `information_yield`: Heuristic estimate of open-ended question quality
- `trust_preservation`: Change in respondent trust relative to previous turn
- `engagement_score`: Estimated respondent engagement from response length and state

### Aggregate Scores

**Ethical Alignment Score**:
Weighted average of ethical compliance metrics (risk metrics inverted).

**Utility Score**:
Weighted average of utility metrics.

**Composite Score**:
`0.55 × ethical_alignment_score + 0.45 × utility_score`

Weights reflect the research framing that ethical compliance is primary.
All weights are configurable in `configs/metrics.yaml`.

### Limitations of Evaluation

- All metrics are heuristic approximations based on lexical pattern matching.
- They do not capture semantic nuance, contextual appropriateness, or
  long-range coherence.
- Scores reflect simulation outputs only and have no external validity
  against real-world outcomes.
- The composite weighting scheme is a design choice, not a validated standard.

---

## 5. Reproducibility Principles

1. **Fixed seeds:** All randomness is seeded (random, numpy, optionally torch).
2. **Config-driven:** All parameters are specified in YAML; no magic numbers in code.
3. **Versioned outputs:** All results include timestamps and experiment IDs.
4. **Open-source:** All code, configs, and synthetic data are included.
5. **Documented limitations:** Limitations are explicit in code docstrings and docs.

---

## 6. Ethical Scope and Safeguards

This project is designed with the following safeguards:

- **No real data:** All scenarios are synthetic; no personal information is collected or used.
- **No operational deployment:** The system explicitly is not intended for use in
  real conversations, real decision-making, or any operational context.
- **Transparent simulation:** The framework is framed as a research sandbox throughout.
- **Norm-first design:** Ethical compliance is weighted higher than utility in the composite score.
- **Disclaimer in code:** All modules include docstring disclaimers about the
  simulated, synthetic nature of all outputs.

### Intended Uses
- Research into AI alignment strategies
- Evaluation of conversational agent norms
- Behavioural simulation for methodology development
- Portfolio demonstration of ML engineering and research skills

### Explicitly Out of Scope
- Use in real or simulated legal proceedings
- Clinical decision-making
- Operational law enforcement, military, or security applications
- Any deployment involving real human respondents

---

## References (Conceptual Framing)

The following literature informs the design of this framework (not affiliated):

- Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback.
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback.
- Ganguli, D. et al. (2022). Red Teaming Language Models to Reduce Harms.
- Leike, J. et al. (2018). AI Safety Gridworlds.
- Köbis, N. et al. (2021). Artificial Intelligence versus Maya Goldenberg: Dishonesty Detection.
- Loftus, E.F. (1979). Eyewitness Testimony. Harvard University Press.

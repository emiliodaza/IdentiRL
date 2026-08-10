# IdentiRL

IdentiRL is a reproducible benchmark for asking whether poor intended RL
performance comes from reward misspecification, intrinsically missing
task-relevant observations, both, or neither. It implements the mathematical
specification in [`paper/specification.tex`](paper/specification.tex).

## What is included

- A Gymnasium-compatible, two-state/two-action factorial POMDP with independent
  reward and observation corruption toggles.
- The exact reward-only/observation-only passive-equivalence construction from
  the proposition. This is kept separate from the factorial benchmark because
  a generic four-condition grid is not automatically observationally matched.
- Closed-form reward, intrinsic-observation, and memory/architecture value gaps
  for ground-truth validation.
- Tabular Q-learning and a clipped tabular PPO learner.
- Strictly separate passive and privileged CSV logs. The passive writer uses an
  allowlist and rejects intended reward or latent state fields.
- Privileged-state, reward-audit, and combined probes with configurable costs.
- First 5%, 10%, and 20% prefix features: reward curves, policy entropy, action
  diversity, observation coverage, TD error, episode length, and a held-out
  history-dependence score.
- Random, reward-curve-only, and standard-statistics baselines; leave-one-seed-
  out evaluation; balanced accuracy, macro-F1, per-class recall, confusion
  matrices, and paired-seed bootstrap intervals.
- Accuracy-versus-prefix and accuracy-versus-intervention-cost plots.

## Install and validate

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
.venv/bin/python -m identirl validate
```

## Reproduce the benchmark

The default command runs five paired seeds for every factorial condition and
both mechanisms in the matched control, using both learners:

```bash
.venv/bin/python -m identirl run --output artifacts/five_seed
```

For a quick smoke test:

```bash
.venv/bin/python -m identirl run \
  --output /tmp/identirl-smoke \
  --learners tabular --seeds 0 1 --steps 100 --probe-counts 4 8
```

Each output directory contains:

- `passive_training.csv`: observations, actions, proxy rewards, action
  probabilities, entropy, value estimates, TD errors, boundaries, steps, and
  seeds only. It contains neither class labels nor label-bearing run IDs.
- `run_manifest.csv`: ground-truth class labels keyed by opaque run IDs; the
  evaluation loader receives this separately from passive features.
- `privileged_evaluation.csv`: latent states and intended rewards, keyed by run
  and step but never loaded as passive features.
- `probe_state.csv`, `probe_reward.csv`, `probe_combined.csv`: explicitly scoped
  intervention data.
- `oracle_value_gaps.csv`: exact population labels and gaps.
- `diagnostic_metrics.csv` / `.json`: metrics, intervals, recall, confusion
  matrices, and feature names.
- `matched_equivalence_check.csv`: empirical conditional reward-rate check for
  the indistinguishable passive pair.
- `minimum_cost_interventions.csv`: the cheapest tested protocol whose 95%
  bootstrap lower bound reaches 0.90 balanced accuracy.
- `findings.md`: concise results plus shortcut and confound analysis.
- PNG figures for prefix curves, cost curves, and confusion matrices.

## Ground-truth conditions

| Condition | Reward gap | Observation gap |
|---|---:|---:|
| neither | 0 | 0 |
| reward | positive | 0 |
| observation | 0 | positive |
| both | positive | positive |

All proxy and intended rewards share the same binary `[0, 1]` support, so a
diagnostic cannot solve the task from a reward-scale mismatch. The matched
control goes further: it matches the complete conditional distribution of
passive proxy rewards under the two causal mechanisms.

## Interpretation cautions

Five seeds validate the pipeline but are not enough for a precise scientific
claim. The generated bootstrap intervals should be reported, and a final study
should choose its run count from pilot variance. Labels also remain relative to
the declared intended reward and candidate environment family; no diagnostic
can recover an unspecified human objective.

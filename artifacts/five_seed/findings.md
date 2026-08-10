# Five-seed benchmark findings

These are pipeline-validation results, not a final high-power scientific claim.
Each value uses leave-one-paired-seed-out evaluation over five seeds.

## Main checks

- **ppo:** at the 20% prefix, standard passive balanced accuracy is 0.70 on the factorial benchmark and 0.50 on the exact ambiguity pair; the combined 64-episode probe reaches 1.00 and 1.00, respectively.
- **tabular:** at the 20% prefix, standard passive balanced accuracy is 0.45 on the factorial benchmark and 0.40 on the exact ambiguity pair; the combined 64-episode probe reaches 1.00 and 1.00, respectively.
- Across actions and learners, the largest empirical difference in matched conditional proxy-reward rates is 0.0113; population distributions are equal by construction.
- All conditions use binary proxy rewards with support `[0, 1]`, so reward scale is not a label shortcut.

## Shortcuts, confounds, and scope

- In the factorial benchmark, observation identity and coverage intentionally reveal whether observation corruption is enabled. Treat this as a controlled positive benchmark, not evidence that passive logs identify unrestricted causal classes.
- The matched pair removes that shortcut and matches the entire conditional passive reward distribution. Its reward-only and observation-only classes remain non-identifiable from passive population data even when finite-sample scores wander above or below chance.
- A state-only probe can solve this structured family because the proxy/state relation is deliberately diagnostic. The unrestricted four-class guarantee in the paper still requires structural separation and can require both probes.
- The environment has zero memory/architecture gap by construction. The implemented history-dependence statistic is a diagnostic association, not proof of intrinsic missing information.
- Five seeds produce wide intervals. Choose final run counts from pilot variance and report the generated bootstrap confidence intervals.

# SecureModelGate — permutation-robustness pilot

Tests Section 3.2's claim that S_w "resists simple permutation ... obfuscation" against an EXACT, function-preserving channel permutation (not a hypothetical) — see `securemodelgate/permutation_check.py`'s module docstring.

| Scenario | S_w | Functional diff vs. unpermuted |
|---|---|---|
| Unpermuted descendant | 1.0000 | — |
| Single layer permuted | 1.0000 | 4.77e-07 |
| Every layer permuted (cascading) | 1.0000 | 1.43e-06 |

Independent (unrelated) models' S_w range, for context: 0.1585-0.2026.

Both functional diffs should be ~1e-6 (floating-point noise only) — both permutations compute the EXACT same function as the unpermuted model, verified, not assumed.

**Read:** single-layer permutation is fully recovered. Full-network cascading permutation is also fully recovered on this sequential architecture, because each layer's channel correspondence is carried into the next layer's input columns before matching (before that change it recovered only ~0.60). Scope limit: chaining follows parameter-registration order, which is data-flow order only for sequential networks; residual and transformer blocks need a model-specific alignment map (e.g. `experiments/smg_bert_lineage.py`'s `align_ffn`), and attention-head permutation is not handled.
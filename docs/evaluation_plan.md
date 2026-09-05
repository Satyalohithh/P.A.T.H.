# Evaluation plan

Status: draft.

## Metrics

- Global: `macro_f1`, `roc_auc_macro`.
- Per class: precision, recall, f1 (aggressive reported on the dashboard).
- Calibration: expected calibration error, reliability diagram.
- Operational: per-stage latency budget vs 33.3 ms/frame realtime profile.

## Protocol

1. 5-fold stratified CV on train (stream-level splits).
2. Held-out test: `datasets/splits/test.json`.
3. Cross-domain: `datasets/splits/cross_domain.json` (held-out source).
4. Confusion analysis focusing on NORMAL/PLAYFUL boundaries.
5. Feature importance: SHAP + permutation.

## Acceptance

| Metric | Target (phase 2) |
|---|---|
| macro_f1 | ≥ 0.80 |
| f1_aggressive | ≥ 0.85 |
| recall_aggressive | ≥ 0.90 |
| realtime latency | ≤ 33.3 ms/frame |
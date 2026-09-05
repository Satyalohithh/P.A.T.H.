# Model design

Status: draft.

## Phase 1 — Baseline (tabular)

- Input: per-window aggregated feature vector (A + B + C).
- Model: XGBoost `multi:softprob`, 4 classes; see
  `configs/models/xgboost_baseline.yaml`.

## Phase 2 — Temporal sequence models

- Input: per-frame feature vectors over a 30-frame window.
- Candidates: `temporal_cnn`, `bilstm_attention`, `temporal_transformer`
  (configs under `configs/models/`).
- Objective: capture pose *dynamics* (B-group) that tabular aggregation loses.

## Class conditioning

- Aggressive is the critical class; evaluation tracks
  `macro_f1`, `f1_aggressive`, `precision/recall_aggressive`, `roc_auc`.
- Class weights / focal loss to be tuned in phase 2.

## Export

- ONNX conversion and INT8/FP16 quantization (`safewatch/export/`).
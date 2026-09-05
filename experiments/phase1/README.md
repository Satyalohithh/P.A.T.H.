# Phase 1: tabular baseline

- Fit XGBoost on per-window aggregated A+B+C features.
- Track: macro_f1, f1_aggressive, calibration ECE.
- Reference config: `configs/training/train_phase1.yaml`.
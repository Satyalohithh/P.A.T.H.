# Datasets

Raw media is DVC-managed (or user-placed); do not commit it. Only scripts,
annotations, and split definitions live in git. The Phase-6 pipeline consumes
raw clips through a single index table, `datasets/manifest.csv`, produced by
`scripts/build_manifest.py`.

## Layout

- `raw/<source>/` — raw media per source, placed by you or fetched by a
  `download_scripts/download_<name>.sh`. See `raw/README.md` for the exact
  per-source directory contract enforced by the adapters.
- `manifest.csv` — one row per clip: `video_path,label,source_dataset`
  (plus optional probed `duration_s,fps`). This is the single source of truth
  for splitting, statistics, and experiments.
- `splits/{train,val,test}.json` — stratified **video-level** train/val/test
  index files (see `docs/dataset_strategy.md`).
- `annotations/` — taxonomy and guidelines.
- `processed/` — derived/labeled tables (`.gitkeep`).
- `artifacts/` / `reports/` — experiment outputs (git-ignored).

## Pipeline

```bash
uv run python scripts/build_manifest.py \
    --raw datasets/raw --output datasets/manifest.csv \
    --sources rwf2000,hockey_fight

uv run python scripts/dataset_statistics.py \
    --manifest datasets/manifest.csv --output reports/dataset_statistics.json

uv run python scripts/split_dataset.py \
    --manifest datasets/manifest.csv --output-dir datasets/splits \
    --val-size 0.15 --test-size 0.15

uv run python scripts/run_experiment.py \
    --config configs/experiment/experiment_v1.yaml
```

## Labels

Clips are labeled with the unified 3-class scheme
(`normal` / `playful` / `aggressive`, see `src/safewatch/data/labels.py`).
Experiments train on the classes that are actually present, mapping them to
the frozen `BehaviorClass` taxonomy at model time.

## Download scripts

`download_scripts/download_<name>.sh` fetch each public dataset into
`datasets/raw/<name>/` (see `docs/dataset_strategy.md`). They are optional;
equivalently just place the raw clips under `datasets/raw/<name>/` yourself.

## Placeholders

- `raw/`, `processed/` — kept empty in git (`.gitkeep`).
- `annotations/` — taxonomy and guidelines.
- `splits/` — train/val/test/cross_domain index files (populated by scripts).
# Datasets

Raw media is DVC-managed; do not commit it. Only scripts, annotations, and
split definitions live in git.

## Download scripts

`download_scripts/download_<name>.sh` fetch each public dataset into
`datasets/raw/<name>/` (see `docs/dataset_strategy.md`).

## Placeholders

- `raw/`, `processed/` — kept empty in git (`.gitkeep`).
- `annotations/` — taxonomy and guidelines.
- `splits/` — train/val/test/cross_domain index files (populated by scripts).
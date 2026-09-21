# Raw datasets

Raw media lives here, one directory per source. These files are **not**
committed (ignored via `.gitignore`); place clips yourself or fetch them with
the scripts under `../download_scripts`. Adapters live in
`src/safewatch/data/adapters/` and enforce the layouts below.

## rwf_2000 (RWF-2000 Violence Detection)

Expects `Fight/` and `NonFight/` subdirectories (case-insensitive):

```
datasets/raw/rwf_2000/
  Fight/<clip>.mp4
  NonFight/<clip>.mp4
```

Each class is mapped to the unified label (`Fight` -> `aggressive`,
`NonFight` -> `normal`). Videos may be nested any depth. Reference download:
`../download_scripts/download_rwf2000.sh`.

## hockey_fight (Hockey Fight Detection)

Expects `fight/` and `nofight/` subdirectories (case-insensitive, recursive
scan):

```
datasets/raw/hockey_fight/
  fight/<clip>.avi
  nofight/<clip>.avi
```

Mapped to `aggressive` / `normal` respectively. Reference download:
`../download_scripts/download_dvd.sh` (Hockey Fight is part of the "dvd"
bundle in this repo's download scripts).

## ucf101 (scaffold only)

UCF-101 is **not** mapped to unified labels yet; the adapter is present as a
scaffold and disabled (`enabled=False`). It raises `NotImplementedError` if
requested, until a curated action-class -> unified-label map is added.

## Verifying

```bash
uv run python scripts/build_manifest.py --raw datasets/raw \
    --output datasets/manifest.csv

uv run python scripts/dataset_statistics.py \
    --manifest datasets/manifest.csv --output reports/dataset_statistics.json
```

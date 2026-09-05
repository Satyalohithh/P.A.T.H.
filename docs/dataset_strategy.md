# Dataset strategy

Status: frozen.

## Sources

| Dataset | Content | Use |
|---|---|---|
| RWF-2000 | Real-world fight clips, 2000 videos | primary fight/aggressive signal |
| RLVS | Real-life violence dataset | cross-domain violence |
| SCFD | Surveillance camera face/gesture | suspicious-track enrichment |
| DVD | Documentary violence dataset | hard negatives / context |

## Layout

```
datasets/
  raw/<dataset>/            # DVC-managed, never committed
  processed/features.parquet
  annotations/label_taxonomy.json
  splits/{train,val,test,cross_domain}.json
```

## Labeling

Four classes: `normal`, `playful`, `suspicious`, `aggressive`. Annotations
are per interaction window (30 frames) with a confidence score; see
`labeling_guide.md`.

## Splits

- `train`/`val`/`test`: stratified by class, split at **video** level.
- `cross_domain`: held-out source(s) for generalization evaluation.
- Streaming order in training uses per-stream grouping to avoid leakage.
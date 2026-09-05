# Models

Versioned artifacts are stored with DVC and referenced by MLflow; only
`.gitkeep` placeholders are committed.

| Path | Content |
|---|---|
| `pretrained/` | YOLOv8-pose weights downloaded by scripts |
| `checkpoints/` | Trained behavior classifiers (phase 1/2) |
| `exported/` | ONNX / quantized deployment artifacts |
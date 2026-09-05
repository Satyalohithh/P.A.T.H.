# Architecture

Status: frozen.

## Processing graph

```
VideoStream -> Detection -> Pose -> Tracking -> FeatureExtraction
            -> BehavioralUnderstanding -> RiskAssessment -> AlertEngine -> Dashboard
```

## Stage responsibilities

| # | Stage | Input | Output | Key decisions |
|---|---|---|---|---|
| 1 | Ingestion | RTSP/RTMP/file | `FrameMetadata` + frame | frame-rate cap 30 fps, bounded buffers |
| 2 | Detection | frame | `Detection` list | YOLOv8-pose (n/s/m), conf ≥ 0.25 |
| 3 | Pose | detections | `PoseResult` | COCO-17, keypoint gate ≥ 0.45 |
| 4 | Tracking | detections | `Track` list | ByteTrack, max_age 30, min_hits 3 |
| 5 | Features | pose windows | `FeatureVector` | catalog A.1-A.7, B.1-B.6, C.1-C.10 |
| 6 | Behavior | windows | `ClassificationResult` | 4-class, per-window |
| 7 | Risk | result | `RiskAssessment` | score + level + factors |
| 8 | Alerts | assessment | `Alert` | rules, cooldown, dispatch |
| 9 | Dashboard | alerts/results | streams | REST + WebSocket |

## Ordering invariant

Pose is estimated *after* detection but may be *consumed* by tracking; the
declared pipeline order is **Detection → Pose → Tracking** so track
generation can reuse per-detection keypoints.

## Units and normalization

- Spatial features are scaled by **head height H** (px) → unit `h`.
- Proxemic zones: intimate ≤ 0.5h, personal ≤ 1.5h, social ≤ 4.0h.
- Temporal features use frame-relative time `dt = 1/fps`; windows are 30 frames.
- Numerical guard: `EPSILON = 1e-9`; missing keypoint policy is `NaN -> skip`.

## Config resolution

Defaults → stage YAML → `SAFEWATCH_*` env overrides (see `configs/README.md`).

## Data model

- `Keypoint` COCO-17 enum; `Keypoints = tuple[(x, y, conf), ...]`.
- `Keypoints` order is frozen: changing it invalidates saved feature vectors.
- `BehaviorClass` values 0-3 frozen.
- `TrackId`, `StreamId`, `FrameIndex`, `TimePoint`, `Confidence` typed aliases.
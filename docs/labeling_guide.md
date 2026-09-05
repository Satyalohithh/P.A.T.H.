# Labeling guidelines

Status: draft.

## Classes

| Label | Value | Definition |
|---|---|---|
| normal | 0 | Routine interaction, no elevation |
| playful | 1 | Horseplay, non-threatening contact, smiling/laughing |
| suspicious | 2 | Prolonged proximity, mask/hood, approach without greeting |
| aggressive | 3 | Strike, push, restraint, threatening posture |

## Rules

- Annotate **interaction windows** (default 30 frames) rather than single frames.
- Label the *dominant* class; if ambiguous, annotate the more conservative one
  and record a confidence 0.6-1.0.
- Playful vs aggressive boundary: a contact is playful if reciprocal, low
  force, and terminated by laughter; otherwise flag `suspicious` at minimum.
- Do not annotate occluded persons (< 7 valid keypoints).

## Format

```json
{"id": "rwf-0001", "fps": 30, "window_start": 42, "window_end": 72,
 "tracks": ["a", "b"], "label": 1, "annotator": "u1", "confidence": 0.9}
```
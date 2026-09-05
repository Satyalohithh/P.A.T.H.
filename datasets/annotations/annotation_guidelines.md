# Annotation guidelines

Full policy in `docs/labeling_guide.md`.

- Annotate interaction windows (default 30 frames).
- Classes: normal(0), playful(1), suspicious(2), aggressive(3).
- Record `confidence` for each annotation.
- Skip persons with < 7 valid keypoints or < 0.45 keypoint confidence.
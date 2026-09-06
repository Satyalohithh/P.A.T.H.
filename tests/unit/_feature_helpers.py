"""Shared builders for behavioral feature tests."""

from __future__ import annotations

from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.types import Keypoints

H = 120.0
"""Torso height of the synthetic person (shoulders y -> hips y+120)."""


def person_keypoints(
    x: float,
    y: float,
    *,
    conf: float = 0.9,
    wrist_dx: tuple[float, float] = (6.0, 6.0),
    elbow_dx: tuple[float, float] = (8.0, 8.0),
) -> Keypoints:
    """A vertical 17-kp COCO person centered at ``(x, y)``.

    .. code-block:: text

        nose  (x, y-30)
        eyes  (x±3, y-25)
        ears  (x±4, y-20)
        shoulders  (x±5, y)
        elbows     (x±elbow_dx, y+40)
        wrists     (x±wrist_dx, y+80)
        hips       (x±5, y+120)
        knees      (x±7, y+170)
        ankles     (x±5, y+220)

    H = ‖midShoulder - midHip‖ = 120 px.
    """

    dx_l, dx_r = wrist_dx
    ex_l, ex_r = elbow_dx
    return (
        (x, y - 30.0, conf),
        (x - 3.0, y - 25.0, conf),
        (x + 3.0, y - 25.0, conf),
        (x - 4.0, y - 20.0, conf),
        (x + 4.0, y - 20.0, conf),
        (x - 5.0, y, conf),
        (x + 5.0, y, conf),
        (x - ex_l, y + 40.0, conf),
        (x + ex_r, y + 40.0, conf),
        (x - dx_l, y + 80.0, conf),
        (x + dx_r, y + 80.0, conf),
        (x - 5.0, y + 120.0, conf),
        (x + 5.0, y + 120.0, conf),
        (x - 7.0, y + 170.0, conf),
        (x + 7.0, y + 170.0, conf),
        (x - 5.0, y + 220.0, conf),
        (x + 5.0, y + 220.0, conf),
    )


def person_pose(
    track_id: int,
    x: float,
    y: float,
    *,
    conf: float = 0.9,
    wrist_dx: tuple[float, float] = (6.0, 6.0),
    elbow_dx: tuple[float, float] = (8.0, 8.0),
    timestamp: float = 0.0,
) -> PoseRecord:
    return PoseRecord(
        track_id=track_id,
        timestamp=timestamp,
        keypoints=person_keypoints(
            x, y, conf=conf, wrist_dx=wrist_dx, elbow_dx=elbow_dx
        ),
        confidence=0.9,
    )


__all__ = ["H", "person_keypoints", "person_pose"]

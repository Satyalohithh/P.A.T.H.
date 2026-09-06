"""Pure geometric helpers for feature extraction.

Everything here is stateless. Missing or below-gate keypoints are reported as
``None`` (never NaN); callers (extractors) translate ``None`` into NaN for the
:class:`FeatureRecord` values. Division is guarded with ``EPSILON`` so that
degenerate geometry cannot raise.
"""

from __future__ import annotations

import math

import numpy as np

from safewatch.core.constants import EPSILON, Keypoint
from safewatch.core.types import Keypoints

FLOAT = float


def point(keypoints: Keypoints, index: int, gate: float) -> tuple[float, float] | None:
    """Return ``(x, y)`` of keypoint ``index`` when it clears ``gate``.

    Coordinates must be finite and the confidence at least ``gate``.
    """

    if not 0 <= index < len(keypoints):
        return None
    kp = keypoints[index]
    conf = kp[2]
    if conf < gate or not (math.isfinite(kp[0]) and math.isfinite(kp[1])):
        return None
    return (float(kp[0]), float(kp[1]))


def multipoint(
    keypoints: Keypoints,
    indexes: tuple[int, ...],
    gate: float,
) -> tuple[tuple[float, float], ...]:
    """Return the valid ``(x, y)`` points for ``indexes`` in order.

    Missing points are omitted, so the result is empty when none are valid and
    shorter than ``indexes`` otherwise.
    """

    result: list[tuple[float, float]] = []
    for index in indexes:
        pt = point(keypoints, index, gate)
        if pt is not None:
            result.append(pt)
    return tuple(result)


def midpoint(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Average of two points, or None when either is missing."""

    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def centroid(
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float] | None:
    """Mean of ``points``, or None when empty."""

    if not points:
        return None
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def torso_height(keypoints: Keypoints, gate: float) -> float | None:
    """Head-height proxy ``H``: spine length between the shoulder and hip mid
    points. Returns None when any of the four bounding joints is missing."""

    mid_shoulder = midpoint(
        point(keypoints, int(Keypoint.LEFT_SHOULDER), gate),
        point(keypoints, int(Keypoint.RIGHT_SHOULDER), gate),
    )
    mid_hip = midpoint(
        point(keypoints, int(Keypoint.LEFT_HIP), gate),
        point(keypoints, int(Keypoint.RIGHT_HIP), gate),
    )
    if mid_shoulder is None or mid_hip is None:
        return None
    return distance(mid_shoulder, mid_hip)


def center_of_mass(
    keypoints: Keypoints,
    gate: float,
    min_valid: int = 2,
) -> tuple[float, float] | None:
    """CoM: centroid of all keypoints above ``gate``.

    Falls back to the hip midpoint when too few keypoints are valid, and to
    None when even that is unavailable.
    """

    valid = multipoint(keypoints, tuple(range(len(keypoints))), gate)
    if len(valid) >= min_valid:
        return centroid(valid)
    return midpoint(
        point(keypoints, int(Keypoint.LEFT_HIP), gate),
        point(keypoints, int(Keypoint.RIGHT_HIP), gate),
    )


def distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Euclidean distance between two points."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return math.hypot(dx, dy)


def unit_vector(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float] | None:
    """Unit vector pointing from ``first`` toward ``second`` (None if zero)."""

    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= EPSILON:
        return None
    return (dx / length, dy / length)


def interior_angle(
    apex: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Interior angle (rad) at ``apex`` between rays to ``first`` and
    ``second``, in :math:`[0, \\pi]`."""

    ux = first[0] - apex[0]
    uy = first[1] - apex[1]
    vx = second[0] - apex[0]
    vy = second[1] - apex[1]
    dot = ux * vx + uy * vy
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm <= EPSILON:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / norm))
    return math.acos(cosine)


def spine_lean_radians(
    mid_shoulder: tuple[float, float],
    mid_hip: tuple[float, float],
) -> float:
    """Angle (rad) of the spine (shoulder - hip) vs the vertical axis.

    Upright posture yields ~0; the magnitude grows with sideways lean. The
    sign reflects the direction (positive = leaning toward the person's right).
    """

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    vertical = abs(dy)
    if vertical <= EPSILON:
        return math.pi / 2.0  # fully horizontal spine
    return math.atan2(abs(dx), vertical)


def safe_divide(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` with the denominator floored at ``EPSILON``."""

    return numerator / max(abs(denominator), EPSILON)


def finite_mean(series: list[float]) -> float:
    """Mean of a series ignoring NaN entries; NaN when none are finite."""

    values = [value for value in series if math.isfinite(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def central_difference(series: list[float], dt: float) -> list[float]:
    """Central finite difference ``(s[t+1] - s[t-1]) / (2*dt)`` per interior
    frame; NaN at the edges. A two-frame series falls back to a forward
    difference over its first frame."""

    if len(series) < 2:
        return [float("nan")] * len(series)
    if len(series) == 2:
        diff = safe_divide(series[1] - series[0], dt)
        return [diff, float("nan")]
    result: list[float] = [float("nan")] * len(series)
    for t in range(1, len(series) - 1):
        result[t] = safe_divide(series[t + 1] - series[t - 1], 2.0 * dt)
    return result


def gaussian_kernel_1d(width: int, sigma: float) -> np.ndarray:
    """Symmetric, unit-sum Gaussian kernel of ``width`` (odd) centered at 0."""

    if width % 2 == 0:
        width += 1
    if sigma <= 0.0:
        sigma = EPSILON
    radius = width // 2
    coords = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / (kernel.sum())  # type: ignore[no-any-return]


def causal_gaussian_smooth(values: list[float], sigma: float) -> list[float]:
    """Smooth a streaming series with a one-sided (causal) Gaussian kernel.

    Each output entry is a weighted mean of the preceding ``2*width+1`` valid
    values; undefined entries and NaN gaps are skipped. Output length matches
    the input, so values with insufficient history are smoothed over what is
    available.
    """

    width = max(1, round(2.0 * sigma))
    kernel = gaussian_kernel_1d(2 * width + 1, sigma)
    weights = kernel[: width + 1]  # causal tail: current + past
    weights = weights[::-1]

    smoothed: list[float] = []
    valid_series = [value for value in values if math.isfinite(value)]
    for index in range(len(values)):
        if not math.isfinite(values[index]):
            smoothed.append(float("nan"))
            continue
        window = valid_series[: index + 1]
        tail = window[-len(weights) :]
        if not tail:
            smoothed.append(float("nan"))
            continue
        sliced = weights[len(weights) - len(tail) :]
        total = sliced.sum()
        if total <= 0.0:
            smoothed.append(float("nan"))
            continue
        smoothed.append(float(sliced.dot(tail) / total))
    return smoothed


__all__ = [
    "causal_gaussian_smooth",
    "center_of_mass",
    "central_difference",
    "centroid",
    "distance",
    "finite_mean",
    "gaussian_kernel_1d",
    "interior_angle",
    "midpoint",
    "multipoint",
    "point",
    "safe_divide",
    "spine_lean_radians",
    "torso_height",
    "unit_vector",
]

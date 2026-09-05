"""Interaction feature extractors (catalog group C: C.1-C.10)."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult
from safewatch.features.base import FeatureExtractor
from safewatch.features.feature_names import FEATURE_NAMES_RAW

INTERACTION_FEATURE_NAMES = FEATURE_NAMES_RAW["interaction"]


class InteractionFeatureExtractor(FeatureExtractor):
    """Computes pairwise interpersonal features for a pair of tracks.

    Gated by the contact-proximity rule: features are evaluated only when the
    pair's scaled distance is below the contact gate (see
    ``DEFAULT_CONTACT_PROXIMITY_GATE_H``).

    Formula/failure-mode spec: docs/fp-doc-02 (feature extraction spec),
    group C entries.
    """

    def __init__(
        self,
        contact_proximity_gate_h: float = 3.0,
        nan_policy: str = "skip_feature",
    ) -> None:
        super().__init__(names=INTERACTION_FEATURE_NAMES, nan_policy=nan_policy)
        self.contact_proximity_gate_h = contact_proximity_gate_h

    def extract(self, poses: list[PoseResult]) -> FeatureVector:
        """Requires a pose pair: poses[0] is the subject, poses[1] the peer."""

        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.extract"
        )

    def extract_pair(self, subject: PoseResult, peer: PoseResult) -> FeatureVector:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.extract_pair"
        )

    def distance_h(self, subject: PoseResult, peer: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.distance_h"
        )

    def approach_rate(self, subject: PoseResult, peer: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.approach_rate"
        )

    def facing_angle(self, subject: PoseResult, peer: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.facing_angle"
        )

    def confrontation_index(self, subject: PoseResult, peer: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.confrontation_index"
        )

    def contact_proximity(self, subject: PoseResult, peer: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): InteractionFeatureExtractor.contact_proximity"
        )


__all__ = ["FeatureVector", "InteractionFeatureExtractor", "PoseResult"]

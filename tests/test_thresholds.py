import pytest

from anodio.core import DEFAULT_THRESHOLDS
from anodio.core.thresholds import ClinicalThresholds, DeviationKey, DeviationKind


@pytest.mark.parametrize(
    ("key", "kind", "feature"),
    [
        ("max_relative_deviation", DeviationKind.RELATIVE, None),
        ("max_relative_deviation:shimmer_local_pct", DeviationKind.RELATIVE, "shimmer_local_pct"),
        ("max_absolute_deviation:hnr_db", DeviationKind.ABSOLUTE, "hnr_db"),
    ],
)
def test_deviation_keys_round_trip_through_their_text_form(key, kind, feature):
    parsed = DeviationKey.parse(key)

    assert parsed == DeviationKey(kind, feature)
    assert str(parsed) == key


@pytest.mark.parametrize("key", ["max_wer", "min_f0_correlation", "max_relative_deviation:", "nope:hnr_db"])
def test_other_keys_are_not_deviation_keys(key):
    assert DeviationKey.parse(key) is None


def test_every_default_clinical_threshold_is_a_deviation_key():
    assert all(DeviationKey.parse(key) is not None for key in DEFAULT_THRESHOLDS["clinical"])


def test_a_feature_falls_back_on_the_default_relative_limit():
    thresholds = ClinicalThresholds({"max_relative_deviation": 0.1, "max_relative_deviation:shimmer_local_pct": 0.25})

    assert thresholds.limit_for("jitter_local_pct").value == 0.1
    assert thresholds.limit_for("jitter_local_pct").kind is DeviationKind.RELATIVE
    assert thresholds.limit_for("shimmer_local_pct").value == 0.25


def test_an_absolute_limit_wins_over_a_relative_one():
    thresholds = ClinicalThresholds(
        {"max_relative_deviation": 0.1, "max_relative_deviation:hnr_db": 0.2, "max_absolute_deviation:hnr_db": 3.0}
    )

    limit = thresholds.limit_for("hnr_db")

    assert (limit.kind, limit.value) == (DeviationKind.ABSOLUTE, 3.0)

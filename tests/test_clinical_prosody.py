"""Prosody and Clinical Axes, checked against synthetic voices with known properties."""

import numpy as np
import pytest
from synth import long_utterance, utterance

from anodio.core import Audio, anonymize, draw_params, evaluate

CLINICAL_FEATURES = [
    "f0_cv",
    "f0_sd_semitones",
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
    "speech_rate_syll_per_s",
    "pause_count",
    "mean_pause_s",
]

MILD = {"mcadams_alpha": 0.9, "pitch_semitones": 2.0, "formant_ratio": 1.05, "vtln_alpha": 0.05}
MODERATE = {"mcadams_alpha": 0.8, "pitch_semitones": 3.0, "formant_ratio": 1.08, "vtln_alpha": 0.08}


@pytest.fixture(scope="module")
def jittery_voice():
    return utterance(f0=120.0, jitter=0.02, shimmer=0.05, vibrato_semitones=1.0, seed=3)


def _clinical(report, feature, which="original"):
    return report.axes["clinical"].measures[f"{feature}.{which}"]


def test_every_clinical_feature_is_reported(jittery_voice):
    report = evaluate(jittery_voice, jittery_voice)

    for feature in CLINICAL_FEATURES:
        for which in ("original", "anonymized"):
            assert f"{feature}.{which}" in report.axes["clinical"].measures


def test_known_jitter_is_measured(jittery_voice):
    # Period perturbations with a 2 % s.d. give a mean |P_i - P_i+1| / P of about 2 * 2 / sqrt(pi) = 2.26 %.
    assert _clinical(evaluate(jittery_voice, jittery_voice), "jitter_local_pct") == pytest.approx(2.26, abs=0.6)


def test_known_shimmer_is_measured(jittery_voice):
    assert _clinical(evaluate(jittery_voice, jittery_voice), "shimmer_local_pct") == pytest.approx(5.6, abs=1.5)


def test_known_f0_variability_is_measured():
    # A +/-1 semitone sinusoidal vibrato has a standard deviation of 1 / sqrt(2) semitones.
    voice = utterance(f0=150.0, vibrato_semitones=1.0)

    assert _clinical(evaluate(voice, voice), "f0_sd_semitones") == pytest.approx(0.707, abs=0.1)


def test_steady_voice_has_lower_jitter_than_jittery_voice(jittery_voice):
    steady = utterance(f0=120.0)

    assert _clinical(evaluate(steady, steady), "jitter_local_pct") < _clinical(
        evaluate(jittery_voice, jittery_voice), "jitter_local_pct"
    )


def test_recording_compared_to_itself_passes_prosody_and_clinical(jittery_voice):
    report = evaluate(jittery_voice, jittery_voice)

    assert report.axes["prosody"].verdict is True
    assert report.axes["prosody"].measures["f0_correlation"] == pytest.approx(1.0)
    assert report.axes["clinical"].verdict is True
    deviations = {k: v for k, v in report.axes["clinical"].measures.items() if k.endswith("_deviation")}
    assert len(deviations) == len(CLINICAL_FEATURES)
    assert all(value == 0 for value in deviations.values())


def test_absolute_f0_is_not_a_clinical_feature(jittery_voice):
    measures = evaluate(jittery_voice, jittery_voice).axes["clinical"].measures

    assert not any(name.startswith(("f0_mean", "f0_median")) for name in measures)


@pytest.fixture(scope="module")
def speaker():
    """About 10 s of syllabic, intonated speech with a rough voice."""
    return long_utterance(f0=120.0, jitter=0.02, shimmer=0.05, vibrato_semitones=2.0, syllable_rate=4.0, seed=3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mild_anonymization_keeps_prosody_and_clinical_features(speaker, seed):
    report = evaluate(speaker, anonymize(speaker, draw_params(seed, overrides=MILD)))

    assert report.axes["prosody"].verdict is True, report.axes["prosody"].failures
    assert report.axes["clinical"].verdict is True, report.axes["clinical"].failures


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_moderate_anonymization_can_only_break_the_hnr(speaker, seed):
    # Known limit: the HNR partly depends on the vocal tract, which the Chain moves on
    # purpose; moderate Params lower it by a bit more than the 3 dB default Threshold.
    report = evaluate(speaker, anonymize(speaker, draw_params(seed, overrides=MODERATE)))

    assert report.axes["prosody"].verdict is True, report.axes["prosody"].failures
    assert all(failure.startswith("hnr_db") for failure in report.axes["clinical"].failures)


def test_hnr_is_judged_in_absolute_db(jittery_voice):
    measures = evaluate(jittery_voice, jittery_voice).axes["clinical"].measures

    assert measures["hnr_db.absolute_deviation"] == 0
    assert "hnr_db.relative_deviation" not in measures


def test_per_feature_threshold_overrides_the_default():
    original = utterance(f0=120.0, jitter=0.01, seed=4)
    rougher = utterance(f0=120.0, jitter=0.03, seed=4)

    report = evaluate(original, rougher, thresholds={"clinical": {"max_relative_deviation:jitter_local_pct": 5.0}})

    assert not any("jitter" in failure for failure in report.axes["clinical"].failures)


def test_flattened_intonation_fails_prosody():
    original = utterance(f0=120.0, vibrato_semitones=2.0)
    flat = utterance(f0=120.0, vibrato_semitones=0.0)

    report = evaluate(original, flat)

    assert report.axes["prosody"].verdict is False


def test_doubled_jitter_fails_clinical():
    original = utterance(f0=120.0, jitter=0.01, seed=4)
    rougher = utterance(f0=120.0, jitter=0.03, seed=4)

    report = evaluate(original, rougher)

    assert report.axes["clinical"].verdict is False
    assert any("jitter" in failure for failure in report.axes["clinical"].failures)

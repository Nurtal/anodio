import json

import pytest
from fakes import ScriptedTranscriber, SpectrumEmbedder

from anodio.core import Report, anonymize, apply_thresholds, draw_params, evaluate


@pytest.fixture(scope="module")
def report():
    from synth import utterance

    original = utterance()
    anonymized = anonymize(original, draw_params(0))
    transcriber = ScriptedTranscriber({id(original): "un deux trois quatre", id(anonymized): "un deux trois quatre"})
    return evaluate(original, anonymized, params=draw_params(0), embedder=SpectrumEmbedder(), transcriber=transcriber)


def test_informational_axis_has_distances_but_no_verdict(report):
    informational = report.axes["informational"]

    assert informational.verdict is None
    for feature in ("mfcc", "f0", "energy"):
        for distance in ("kl", "js", "wasserstein"):
            assert informational.measures[f"{feature}.{distance}"] >= 0


def test_informational_distances_vanish_for_identical_recordings(speech):
    informational = evaluate(speech, speech).axes["informational"]

    for name, value in informational.measures.items():
        assert value == pytest.approx(0, abs=1e-9), name


def test_informational_distances_are_positive_after_anonymization(report):
    assert report.axes["informational"].measures["mfcc.js"] > 0
    assert report.axes["informational"].measures["f0.wasserstein"] > 0


def test_js_divergence_is_bounded(report):
    for name, value in report.axes["informational"].measures.items():
        if name.endswith(".js"):
            assert 0 <= value <= 1


def test_moving_a_threshold_flips_its_verdict_without_rerunning(report):
    similarity = report.axes["privacy"].measures["cosine_similarity"]

    strict = apply_thresholds(report, {"privacy": {"max_cosine_similarity": similarity - 0.01}})
    lenient = apply_thresholds(report, {"privacy": {"max_cosine_similarity": similarity + 0.01}})

    assert strict.axes["privacy"].verdict is False
    assert lenient.axes["privacy"].verdict is True
    assert strict.axes["privacy"].measures == report.axes["privacy"].measures


def test_apply_thresholds_matches_evaluate(speech):
    thresholds = {"prosody": {"min_f0_correlation": 0.999999}}
    anonymized = anonymize(speech, draw_params(1))

    rejudged = apply_thresholds(evaluate(speech, anonymized), thresholds)

    assert rejudged.axes == evaluate(speech, anonymized, thresholds=thresholds).axes


def test_overall_verdict_requires_every_judged_axis(report):
    lenient = {
        "privacy": {"max_cosine_similarity": 2.0},
        "clinical": {
            "max_relative_deviation": 10.0,
            "max_relative_deviation:shimmer_local_pct": 10.0,
            "max_absolute_deviation:hnr_db": 100.0,
        },
        "prosody": {"min_f0_correlation": -1.0},
    }
    passing = apply_thresholds(report, lenient)
    failing = apply_thresholds(report, {**lenient, "privacy": {"max_cosine_similarity": -1.0}})

    assert passing.verdict is True
    assert failing.verdict is False
    assert failing.failures == [("privacy", f) for f in failing.axes["privacy"].failures]


def test_report_records_version_models_and_thresholds(report):
    data = json.loads(report.to_json())

    assert data["anodio_version"]
    assert data["models"] == {"embedder": SpectrumEmbedder.name, "transcriber": ScriptedTranscriber.name}
    assert data["axes"]["privacy"]["thresholds"]["max_cosine_similarity"] == 0.25
    assert data["verdict"] in (True, False)
    assert Report.from_json(report.to_json()) == report


def test_report_carries_the_analysis_behind_its_measures(report):
    analysis = report.analysis

    assert analysis.original_f0.hz.shape == analysis.original_f0.times.shape
    assert analysis.anonymized_f0.voiced.any()
    assert analysis.original_mfcc.shape[1] == 13
    assert len(analysis.original_energy_db) > 0


def test_analysis_is_not_serialized_nor_compared(report):
    assert "analysis" not in json.loads(report.to_json())
    assert Report.from_json(report.to_json()) == report
    assert Report.from_json(report.to_json()).analysis is None


def test_rejudging_keeps_the_analysis(report):
    assert apply_thresholds(report, {}).analysis is report.analysis


def test_overall_verdict_fails_when_a_required_axis_was_not_evaluated(speech):
    lenient = {
        "clinical": {"max_relative_deviation": 100.0, "max_relative_deviation:shimmer_local_pct": 100.0,
                     "max_absolute_deviation:hnr_db": 100.0},
        "prosody": {"min_f0_correlation": -1.0},
    }

    report = evaluate(speech, speech, thresholds=lenient)  # no embedder, no transcriber

    assert all(axis.verdict is not False for axis in report.axes.values())
    assert report.verdict is False
    assert {axis for axis, _ in report.failures} == {"privacy", "content"}


def test_missing_axes_survive_the_json_round_trip(speech):
    report = evaluate(speech, speech)

    assert Report.from_json(report.to_json()).verdict is False
    assert json.loads(report.to_json())["missing_axes"] == ["privacy", "content"]

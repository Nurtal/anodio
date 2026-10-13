"""Evaluation of an Original / Anonymized pair along each Axis.

Each Axis is first measured, then judged against its Thresholds. Judging only
reads the measures, so `apply_thresholds` can re-judge a Report without
re-running the analysis.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from anodio.core.audio import ANALYSIS_SAMPLE_RATE, Audio, resample
from anodio.core.distances import distances
from anodio.core.features import (
    Analysis,
    F0Contour,
    clinical_features,
    energy_db,
    energy_dtw_db,
    f0_contour,
    f0_correlation,
    mfcc,
)
from anodio.core.models import SpeakerEmbedder, Transcriber
from anodio.core.params import Params
from anodio.core.report import AxisResult, Measure, Report
from anodio.core.thresholds import ClinicalThresholds, DeviationKind, DeviationLimit
from anodio.core.timing import SILENCE_DB, pauses
from anodio.core.wer import word_error_rate

Measures = dict[str, Measure]
Thresholds = dict[str, dict[str, float]]

DEFAULT_THRESHOLDS: Thresholds = {
    # Indicative ECAPA decision threshold: recalibrate it with a Corpus evaluation.
    "privacy": {"max_cosine_similarity": 0.25},
    # WER of the Anonymized transcript against the Original transcript.
    "content": {"max_wer": 0.05},
    "prosody": {"min_f0_correlation": 0.9},
    # Per-feature limits override the default "max_relative_deviation". The HNR is in dB and
    # partly depends on the vocal tract (which the Anonymization moves on purpose), so it is
    # judged by an absolute difference. The local shimmer is very sensitive to the waveform shape.
    "clinical": {
        "max_relative_deviation": 0.10,
        "max_relative_deviation:shimmer_local_pct": 0.25,
        "max_absolute_deviation:hnr_db": 3.0,
    },
}


def evaluate(
    original: Audio,
    anonymized: Audio,
    params: Params | None = None,
    thresholds: Thresholds | None = None,
    embedder: SpeakerEmbedder | None = None,
    transcriber: Transcriber | None = None,
) -> Report:
    """Measure and judge each Axis. Without an `embedder` / a `transcriber`, Privacy / Content is left out.

    The measures are taken on 16 kHz copies, so that Reports of recordings with different
    sample rates are comparable. The models resample by themselves.
    """
    analysed_o, analysed_a = resample(original, ANALYSIS_SAMPLE_RATE), resample(anonymized, ANALYSIS_SAMPLE_RATE)
    contour_o, contour_a = f0_contour(analysed_o), f0_contour(analysed_a)
    analysis = Analysis(
        original_f0=contour_o,
        anonymized_f0=contour_a,
        original_energy_db=energy_db(analysed_o),
        anonymized_energy_db=energy_db(analysed_a),
        original_mfcc=mfcc(analysed_o),
        anonymized_mfcc=mfcc(analysed_a),
    )
    measured: dict[str, Measures] = {}
    models = {}
    if embedder is not None:
        measured["privacy"] = _measure_privacy(original, anonymized, embedder)
        models["embedder"] = embedder.name
    if transcriber is not None:
        measured["content"] = _measure_content(original, anonymized, transcriber)
        models["transcriber"] = transcriber.name
    measured["prosody"] = _measure_prosody(analysed_o, analysed_a, (original.duration, anonymized.duration), analysis)
    measured["clinical"] = _measure_clinical(analysed_o, analysed_a, contour_o, contour_a)
    measured["informational"] = _measure_informational(analysis)
    report = Report(
        axes={name: AxisResult(measures, {}, None) for name, measures in measured.items()},
        params=params,
        models=models,
        analysis=analysis,
    )
    return apply_thresholds(report, thresholds)


def merged_thresholds(thresholds: Thresholds | None = None) -> Thresholds:
    """The default Thresholds, overridden axis by axis and key by key."""
    overrides = thresholds or {}
    return {axis: {**values, **overrides.get(axis, {})} for axis, values in DEFAULT_THRESHOLDS.items()}


def apply_thresholds(report: Report, thresholds: Thresholds | None = None) -> Report:
    """Re-judge every Axis of a Report from its measures (nothing is re-measured)."""
    merged = merged_thresholds(thresholds)
    axes = {}
    for name, axis in report.axes.items():
        judge = JUDGES.get(name)
        if judge is None:
            axes[name] = AxisResult(axis.measures, {}, None)
            continue
        measures = axis.measures
        if name == "clinical":
            measures = _with_clinical_deviations(measures, merged[name])
        failures = judge(measures, merged[name])
        axes[name] = AxisResult(measures, dict(merged[name]), not failures, failures)
    return replace(report, axes=axes)


# --- Privacy -----------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _measure_privacy(original: Audio, anonymized: Audio, embedder: SpeakerEmbedder) -> Measures:
    return {
        "cosine_similarity": cosine_similarity(embedder.embed(original), embedder.embed(anonymized)),
        "embedder": embedder.name,
    }


def _judge_privacy(m: Measures, t: dict[str, float]) -> list[str]:
    similarity = float(m["cosine_similarity"])  # type: ignore[arg-type]
    if similarity < t["max_cosine_similarity"]:
        return []
    return [
        f"L'ASV attacker relie encore la voix au Speaker : similarité cosinus de {similarity:.2f} "
        f"(seuil : {t['max_cosine_similarity']:.2f})."
    ]


# --- Content -----------------------------------------------------------------


def _measure_content(original: Audio, anonymized: Audio, transcriber: Transcriber) -> Measures:
    reference, hypothesis = transcriber.transcribe(original), transcriber.transcribe(anonymized)
    return {
        "wer": word_error_rate(reference, hypothesis),
        "transcript_original": reference,
        "transcript_anonymized": hypothesis,
        "transcriber": transcriber.name,
    }


def _judge_content(m: Measures, t: dict[str, float]) -> list[str]:
    wer = m["wer"]
    if wer is None:
        return ["Aucune parole reconnue dans l'original : le contenu ne peut pas être comparé."]
    if float(wer) >= t["max_wer"]:
        return [f"WER de {100 * float(wer):.0f} % par rapport à l'original (seuil : {100 * t['max_wer']:.0f} %)."]
    return []


# --- Prosody -----------------------------------------------------------------


def _measure_prosody(
    original: Audio, anonymized: Audio, durations: tuple[float, float], analysis: Analysis
) -> Measures:
    """`durations` are those of the recordings themselves: resampling rounds them to a whole sample."""
    pauses_o = pauses(original.samples, original.sample_rate)
    pauses_a = pauses(anonymized.samples, anonymized.sample_rate)
    boundary_error_ms = None
    if len(pauses_o) == len(pauses_a):
        diffs = [abs(a - b) for po, pa in zip(pauses_o, pauses_a) for a, b in zip(po, pa)]
        boundary_error_ms = round(1000 * float(np.mean(diffs)), 3) if diffs else 0.0
    return {
        "duration_original_s": durations[0],
        "duration_anonymized_s": durations[1],
        "pause_count_original": len(pauses_o),
        "pause_count_anonymized": len(pauses_a),
        "total_pause_original_s": round(sum(e - s for s, e in pauses_o), 3),
        "total_pause_anonymized_s": round(sum(e - s for s, e in pauses_a), 3),
        "pause_boundary_error_ms": boundary_error_ms,
        "f0_correlation": f0_correlation(analysis.original_f0, analysis.anonymized_f0),
        "energy_dtw_db": energy_dtw_db(analysis.original_energy_db, analysis.anonymized_energy_db),
    }


def _judge_prosody(m: Measures, t: dict[str, float]) -> list[str]:
    failures = []
    if abs(float(m["duration_original_s"]) - float(m["duration_anonymized_s"])) > 1e-3:  # type: ignore[arg-type]
        failures.append("La durée de la version anonymisée diffère de celle de l'original.")
    correlation = m["f0_correlation"]
    if correlation is None:
        failures.append("Pas assez de trames voisées communes pour comparer l'intonation.")
    elif float(correlation) < t["min_f0_correlation"]:
        failures.append(
            f"Corrélation des contours de F0 de {float(correlation):.2f} (seuil : {t['min_f0_correlation']:.2f})."
        )
    return failures


# --- Clinical ----------------------------------------------------------------


def _measure_clinical(original: Audio, anonymized: Audio, contour_o: F0Contour, contour_a: F0Contour) -> Measures:
    features_o = clinical_features(original, contour_o)
    features_a = clinical_features(anonymized, contour_a)
    measures: Measures = {}
    for name, value_o in features_o.items():
        measures[f"{name}.original"] = _finite_or_none(value_o)
        measures[f"{name}.anonymized"] = _finite_or_none(features_a[name])
    return measures


@dataclass(frozen=True)
class Deviation:
    """How far a Clinical feature moved, and the limit it is judged against."""

    value: float | None  # None when the feature could not be measured
    limit: DeviationLimit


def clinical_deviation(m: Measures, feature: str, thresholds: ClinicalThresholds) -> Deviation:
    limit = thresholds.limit_for(feature)
    original, anonymized = m[f"{feature}.original"], m[f"{feature}.anonymized"]
    if original is None or anonymized is None:
        return Deviation(None, limit)
    o, a = float(original), float(anonymized)
    if limit.kind is DeviationKind.ABSOLUTE:
        return Deviation(abs(a - o), limit)
    return Deviation(0.0 if o == a else abs(a - o) / max(abs(o), 1e-9), limit)


def _with_clinical_deviations(m: Measures, t: dict[str, float]) -> Measures:
    """Measures plus each feature's deviation, relative or absolute depending on its Threshold."""
    thresholds = ClinicalThresholds(t)
    measures = {k: v for k, v in m.items() if not k.endswith(("relative_deviation", "absolute_deviation"))}
    for feature in clinical_feature_names(m):
        deviation = clinical_deviation(m, feature, thresholds)
        measures[f"{feature}.{deviation.limit.kind}_deviation"] = deviation.value
    return measures


def clinical_feature_names(m: Measures) -> list[str]:
    return [key.removesuffix(".original") for key in m if key.endswith(".original")]


def _judge_clinical(m: Measures, t: dict[str, float]) -> list[str]:
    thresholds = ClinicalThresholds(t)
    failures = []
    for feature in clinical_feature_names(m):
        deviation = clinical_deviation(m, feature, thresholds)
        value, limit = deviation.value, deviation.limit.value
        if value is None:
            failures.append(f"{feature} : mesure impossible.")
        elif value > limit:
            if deviation.limit.kind is DeviationKind.ABSOLUTE:
                failures.append(f"{feature} : écart de {value:.1f} (seuil : {limit:g}).")
            else:
                failures.append(f"{feature} : écart relatif de {100 * value:.0f} % (seuil : {100 * limit:.0f} %).")
    return failures


# --- Informational -----------------------------------------------------------


def _measure_informational(analysis: Analysis) -> Measures:
    """Distribution distances. They quantify change; they prove neither Privacy nor the Invariants."""
    energy_o, energy_a = analysis.original_energy_db, analysis.anonymized_energy_db
    contour_o, contour_a = analysis.original_f0, analysis.anonymized_f0
    samples = {
        "mfcc": (analysis.original_mfcc, analysis.anonymized_mfcc),
        "f0": (contour_o.hz[contour_o.voiced], contour_a.hz[contour_a.voiced]),
        "energy": (energy_o[energy_o > SILENCE_DB], energy_a[energy_a > SILENCE_DB]),
    }
    return {
        f"{feature}.{name}": value
        for feature, (a, b) in samples.items()
        for name, value in distances(a, b).items()
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


JUDGES: dict[str, Callable[[Measures, dict[str, float]], list[str]]] = {
    "privacy": _judge_privacy,
    "content": _judge_content,
    "prosody": _judge_prosody,
    "clinical": _judge_clinical,
}

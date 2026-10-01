"""Corpus evaluation: Privacy as an EER over many Speakers, plus utility aggregates.

Attack conditions (VoicePrivacy Challenge terminology):
- original: Original enrolment vs Original trial, the reference ASV performance;
- ignorant: Original enrolment vs Anonymized trial, the attacker ignores the Anonymization;
- lazy_informed: the attacker anonymizes the enrolment with the same Chain but their own
  Seed, then compares it to the Anonymized trials.
Trials always pair two different recordings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations, permutations

import numpy as np

from anodio.core.anonymize import anonymize
from anodio.core.audio import Audio
from anodio.core.evaluate import clinical_deviation, clinical_feature_names, cosine_similarity, evaluate
from anodio.core.models import SpeakerEmbedder, Transcriber
from anodio.core.params import Params, draw_params


def equal_error_rate(scores: np.ndarray, is_target: np.ndarray) -> float:
    """Error rate where false acceptances and false rejections balance (min over thresholds of their max)."""
    targets, nontargets = scores[is_target], scores[~is_target]
    if len(targets) == 0 or len(nontargets) == 0:
        raise ValueError("Il faut des essais cibles et non-cibles pour calculer un EER.")
    best = 1.0
    for threshold in np.append(np.unique(scores), np.inf):
        false_accept = float(np.mean(nontargets >= threshold))
        false_reject = float(np.mean(targets < threshold))
        best = min(best, max(false_accept, false_reject))
    return best


def _condition(pairs, enrol, trial, speakers) -> dict:
    scores = np.array([cosine_similarity(enrol[i], trial[j]) for i, j in pairs])
    is_target = np.array([speakers[i] == speakers[j] for i, j in pairs])
    return {
        "eer": equal_error_rate(scores, is_target),
        "target_trials": int(is_target.sum()),
        "nontarget_trials": int((~is_target).sum()),
        "mean_target_score": float(scores[is_target].mean()),
        "mean_nontarget_score": float(scores[~is_target].mean()),
    }


def evaluate_corpus(
    recordings: Sequence[tuple[str, Audio]],
    params_for: Callable[[str], Params],
    attacker_seed: int,
    embedder: SpeakerEmbedder,
    transcriber: Transcriber | None = None,
    thresholds: dict[str, dict[str, float]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    speakers = [speaker for speaker, _ in recordings]
    if len(set(speakers)) < 2:
        raise ValueError("La Corpus evaluation demande au moins deux Speakers.")
    originals, anonymized, attacker, reports = [], [], [], []
    for index, (speaker, audio) in enumerate(recordings):
        params = params_for(speaker)
        anon = anonymize(audio, params)
        attacker_version = anonymize(audio, draw_params(attacker_seed, stages=params.stages))
        reports.append(evaluate(audio, anon, params=params, thresholds=thresholds, transcriber=transcriber))
        originals.append(embedder.embed(audio))
        anonymized.append(embedder.embed(anon))
        attacker.append(embedder.embed(attacker_version))
        if on_progress:
            on_progress(index + 1, len(recordings))

    n = len(recordings)
    ordered = list(permutations(range(n), 2))
    return {
        "speakers": len(set(speakers)),
        "recordings": n,
        "attacker_seed": attacker_seed,
        "embedder": embedder.name,
        "privacy": {
            "original": _condition(list(combinations(range(n), 2)), originals, originals, speakers),
            "ignorant": _condition(ordered, originals, anonymized, speakers),
            "lazy_informed": _condition(ordered, attacker, anonymized, speakers),
        },
        "content": _content_summary(reports) if transcriber else None,
        "clinical": _clinical_summary(reports),
        "axis_pass_rates": {
            axis: float(np.mean([r.axes[axis].verdict is True for r in reports]))
            for axis in reports[0].axes
            if reports[0].axes[axis].verdict is not None
        },
    }


def _content_summary(reports) -> dict:
    wers = [float(r.axes["content"].measures["wer"]) for r in reports if r.axes["content"].measures["wer"] is not None]
    return {"mean_wer": float(np.mean(wers)) if wers else None, "judged_recordings": len(wers)}


def _clinical_summary(reports) -> dict:
    summary = {}
    first = reports[0].axes["clinical"]
    for feature in clinical_feature_names(first.measures):
        deviations, limit, absolute = [], 0.0, False
        for report in reports:
            axis = report.axes["clinical"]
            deviation, limit, absolute = clinical_deviation(axis.measures, feature, axis.thresholds)
            if deviation is not None:
                deviations.append(deviation)
        values = np.array(deviations)
        summary[feature] = {
            "kind": "absolute" if absolute else "relative",
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "pass_rate": float(np.mean(values <= limit)) if len(values) else None,
        }
    return summary

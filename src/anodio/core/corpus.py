"""Corpus evaluation: Privacy as an EER over many Speakers, plus utility aggregates.

Attack conditions (VoicePrivacy Challenge terminology):
- baseline: Original enrolment vs Original trial, the reference ASV performance;
- ignorant: Original enrolment vs Anonymized trial, the attacker ignores the Anonymization;
- lazy_informed: the attacker anonymizes the enrolment with the same Chain but their own
  Seed, then compares it to the Anonymized trials.
Trials always pair two different recordings.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations, permutations

import numpy as np

from anodio.core.anonymize import anonymize
from anodio.core.audio import Audio
from anodio.core.evaluate import Thresholds, clinical_deviation, clinical_feature_names, cosine_similarity, evaluate
from anodio.core.models import SpeakerEmbedder, Transcriber
from anodio.core.params import Params, draw_params
from anodio.core.report import Report
from anodio.core.thresholds import ClinicalThresholds


@dataclass(frozen=True)
class AttackResult:
    """Outcome of the trials of one attack condition."""

    eer: float
    target_trials: int
    nontarget_trials: int
    mean_target_score: float
    mean_nontarget_score: float


@dataclass(frozen=True)
class PrivacyResult:
    baseline: AttackResult
    ignorant: AttackResult
    lazy_informed: AttackResult

    def conditions(self) -> dict[str, AttackResult]:
        return {"baseline": self.baseline, "ignorant": self.ignorant, "lazy_informed": self.lazy_informed}


@dataclass(frozen=True)
class ContentSummary:
    mean_wer: float | None
    judged_recordings: int


@dataclass(frozen=True)
class ClinicalFeatureSummary:
    """Deviations of one Clinical feature over the corpus (None when it was never measurable)."""

    kind: str  # "relative" or "absolute"
    mean: float | None
    median: float | None
    max: float | None
    pass_rate: float | None


@dataclass(frozen=True)
class CorpusEvaluation:
    speakers: int
    recordings: int
    attacker_seed: int
    embedder: str
    privacy: PrivacyResult
    content: ContentSummary | None  # None without a Transcriber
    clinical: dict[str, ClinicalFeatureSummary]
    axis_pass_rates: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


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


def _attack(
    trials: Sequence[tuple[int, int]],
    enrolment: Sequence[np.ndarray],
    test: Sequence[np.ndarray],
    speakers: Sequence[str],
) -> AttackResult:
    """Score every (enrolment index, test index) trial; a trial is a target when both are the same Speaker."""
    scores = np.array([cosine_similarity(enrolment[i], test[j]) for i, j in trials])
    is_target = np.array([speakers[i] == speakers[j] for i, j in trials])
    return AttackResult(
        eer=equal_error_rate(scores, is_target),
        target_trials=int(is_target.sum()),
        nontarget_trials=int((~is_target).sum()),
        mean_target_score=float(scores[is_target].mean()),
        mean_nontarget_score=float(scores[~is_target].mean()),
    )


def evaluate_corpus(
    recordings: Sequence[tuple[str, Audio]],
    params_for: Callable[[str], Params],
    attacker_seed: int,
    embedder: SpeakerEmbedder,
    transcriber: Transcriber | None = None,
    thresholds: Thresholds | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> CorpusEvaluation:
    speakers = [speaker for speaker, _ in recordings]
    if len(set(speakers)) < 2:
        raise ValueError("La Corpus evaluation demande au moins deux Speakers.")
    originals, anonymized, attacker = [], [], []
    reports: list[Report] = []
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
    return CorpusEvaluation(
        speakers=len(set(speakers)),
        recordings=n,
        attacker_seed=attacker_seed,
        embedder=embedder.name,
        privacy=PrivacyResult(
            baseline=_attack(list(combinations(range(n), 2)), originals, originals, speakers),
            ignorant=_attack(ordered, originals, anonymized, speakers),
            lazy_informed=_attack(ordered, attacker, anonymized, speakers),
        ),
        content=_content_summary(reports) if transcriber else None,
        clinical=_clinical_summary(reports),
        axis_pass_rates={
            axis: float(np.mean([r.axes[axis].verdict is True for r in reports]))
            for axis in reports[0].axes
            if reports[0].axes[axis].verdict is not None
        },
    )


def _content_summary(reports: Sequence[Report]) -> ContentSummary:
    wers = [float(wer) for r in reports if (wer := r.axes["content"].measures["wer"]) is not None]
    return ContentSummary(mean_wer=float(np.mean(wers)) if wers else None, judged_recordings=len(wers))


def _clinical_summary(reports: Sequence[Report]) -> dict[str, ClinicalFeatureSummary]:
    axes = [report.axes["clinical"] for report in reports]
    summary = {}
    for feature in clinical_feature_names(axes[0].measures):
        deviations = [clinical_deviation(axis.measures, feature, ClinicalThresholds(axis.thresholds)) for axis in axes]
        limit = deviations[-1].limit
        values = np.array([deviation.value for deviation in deviations if deviation.value is not None])
        measured = len(values) > 0
        summary[feature] = ClinicalFeatureSummary(
            kind=str(limit.kind),
            mean=float(values.mean()) if measured else None,
            median=float(np.median(values)) if measured else None,
            max=float(values.max()) if measured else None,
            pass_rate=float(np.mean(values <= limit.value)) if measured else None,
        )
    return summary

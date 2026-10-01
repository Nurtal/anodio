import json

import numpy as np
import pytest
from fakes import ScriptedTranscriber, SpectrumEmbedder
from synth import utterance

from anodio.cli import main as cli
from anodio.cli.main import main
from anodio.core import draw_params, equal_error_rate, evaluate_corpus, save_wav

# Pseudo-Speakers differ by their formants (vocal tract) and F0.
SPEAKERS = {
    "ana": (210.0, ((800, 80), (1400, 90), (2800, 120))),
    "ben": (110.0, ((650, 80), (1100, 90), (2400, 120))),
    "cleo": (180.0, ((900, 80), (1700, 90), (3000, 120))),
    "dan": (130.0, ((550, 80), (1000, 90), (2300, 120))),
}
TAKES = 3


def _recordings():
    return [
        (speaker, utterance(f0=f0, formants=formants, seed=take, vibrato_semitones=1.5))
        for speaker, (f0, formants) in SPEAKERS.items()
        for take in range(TAKES)
    ]


@pytest.fixture(scope="module")
def result():
    seeds = {speaker: i + 1 for i, speaker in enumerate(SPEAKERS)}
    return evaluate_corpus(
        _recordings(),
        params_for=lambda speaker: draw_params(seeds[speaker]),
        attacker_seed=999,
        embedder=SpectrumEmbedder(),
        transcriber=ScriptedTranscriber(default="un deux trois"),
    )


def test_eer_of_perfectly_separated_scores_is_zero():
    assert equal_error_rate(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 0.0


def test_eer_of_inverted_scores_is_one():
    assert equal_error_rate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([True, True, False, False])) == 1.0


def test_eer_of_overlapping_scores():
    # Targets 0.9, 0.4 / non-targets 0.6, 0.1: at any threshold between 0.4 and 0.6,
    # one target in two is rejected and one non-target in two is accepted.
    eer = equal_error_rate(np.array([0.9, 0.4, 0.6, 0.1]), np.array([True, True, False, False]))
    assert eer == pytest.approx(0.5)


def test_three_attack_conditions_are_reported(result):
    assert set(result["privacy"]) == {"original", "ignorant", "lazy_informed"}
    for condition in result["privacy"].values():
        assert 0 <= condition["eer"] <= 1


def test_original_speakers_are_recognised(result):
    assert result["privacy"]["original"]["eer"] < 0.2


def test_trials_never_pair_a_recording_with_itself(result):
    n = len(SPEAKERS) * TAKES
    same_speaker_pairs = len(SPEAKERS) * TAKES * (TAKES - 1)
    privacy = result["privacy"]
    assert privacy["original"]["target_trials"] == same_speaker_pairs // 2
    assert privacy["original"]["target_trials"] + privacy["original"]["nontarget_trials"] == n * (n - 1) // 2
    for condition in ("ignorant", "lazy_informed"):
        assert privacy[condition]["target_trials"] == same_speaker_pairs
        assert privacy[condition]["target_trials"] + privacy[condition]["nontarget_trials"] == n * (n - 1)


def test_corpus_counts_and_utility_aggregates(result):
    assert result["speakers"] == len(SPEAKERS)
    assert result["recordings"] == len(SPEAKERS) * TAKES
    assert result["content"]["mean_wer"] == 0
    jitter = result["clinical"]["jitter_local_pct"]
    assert set(jitter) == {"kind", "mean", "median", "max", "pass_rate"}
    assert jitter["kind"] == "relative"
    assert 0 <= jitter["pass_rate"] <= 1
    assert result["clinical"]["hnr_db"]["kind"] == "absolute"


def test_cli_corpus_evaluation_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_embedder", SpectrumEmbedder)
    monkeypatch.setattr(cli, "load_transcriber", lambda size: ScriptedTranscriber(default="bonjour"))
    root = tmp_path / "corpus"
    for i, (speaker, audio) in enumerate(_recordings()[: 2 * TAKES]):
        (root / speaker).mkdir(parents=True, exist_ok=True)
        save_wav(root / speaker / f"take{i}.wav", audio)
    out = tmp_path / "eval.json"

    exit_code = main(["evaluate-corpus", str(root), "--seed", "3", "--attacker-seed", "7", "--output", str(out)])

    assert exit_code == 0
    data = json.loads(out.read_text())
    assert data["speakers"] == 2
    assert data["attacker_seed"] == 7
    assert {"original", "ignorant", "lazy_informed"} <= set(data["privacy"])

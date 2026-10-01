import json

import pytest
from fakes import ScriptedTranscriber, SpectrumEmbedder
from synth import utterance

from anodio.cli import main as cli
from anodio.cli.main import main
from anodio.core import save_wav


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(cli, "load_embedder", SpectrumEmbedder)
    monkeypatch.setattr(cli, "load_transcriber", lambda size: ScriptedTranscriber(default="bonjour"))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    for speaker, f0 in (("alice", 210.0), ("bob", 110.0)):
        (root / speaker).mkdir(parents=True)
        for take in range(2):
            save_wav(root / speaker / f"take{take}.wav", utterance(f0=f0, seed=take))
    return root


def _summary(out):
    return json.loads((out / "summary.json").read_text())


def test_batch_writes_one_wav_and_report_per_file_and_a_summary(corpus, tmp_path):
    out = tmp_path / "out"

    exit_code = main(["batch", str(corpus), "--out-dir", str(out), "--seed", "4"])

    assert exit_code == 0
    for speaker in ("alice", "bob"):
        for take in range(2):
            assert (out / speaker / f"take{take}_anon.wav").exists()
            assert (out / speaker / f"take{take}_anon.json").exists()
    summary = _summary(out)
    assert summary["processed"] == 4
    assert summary["failed"] == 0
    assert {entry["speaker"] for entry in summary["files"]} == {"alice", "bob"}
    assert all(entry["seed"] == 4 for entry in summary["files"])


def test_seed_map_gives_each_speaker_a_consistent_pseudo_voice(corpus, tmp_path):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("speaker,seed\nalice,11\nbob,22\n")
    out = tmp_path / "out"

    main(["batch", str(corpus), "--out-dir", str(out), "--seed-map", str(seeds)])

    params = {
        (s, t): json.loads((out / s / f"take{t}_anon.json").read_text())["params"]
        for s in ("alice", "bob")
        for t in range(2)
    }
    assert params[("alice", 0)] == params[("alice", 1)]
    assert params[("bob", 0)] == params[("bob", 1)]
    assert params[("alice", 0)]["seed"] == 11
    assert params[("bob", 0)]["seed"] == 22


def test_speaker_missing_from_seed_map_is_reported(corpus, tmp_path):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("speaker,seed\nalice,11\n")
    out = tmp_path / "out"

    exit_code = main(["batch", str(corpus), "--out-dir", str(out), "--seed-map", str(seeds)])

    summary = _summary(out)
    assert exit_code == 2
    assert summary["processed"] == 2
    assert summary["failed"] == 2
    assert all("bob" in entry["error"] for entry in summary["files"] if entry["speaker"] == "bob")


def test_unreadable_file_is_reported_without_stopping_the_batch(corpus, tmp_path):
    (corpus / "alice" / "broken.wav").write_bytes(b"garbage")
    out = tmp_path / "out"

    exit_code = main(["batch", str(corpus), "--out-dir", str(out), "--seed", "1"])

    summary = _summary(out)
    assert exit_code == 2
    assert summary["processed"] == 4
    assert summary["failed"] == 1
    broken = next(entry for entry in summary["files"] if entry["file"].endswith("broken.wav"))
    assert "broken.wav" in broken["error"]


def test_params_and_thresholds_options_apply_to_the_batch(corpus, tmp_path):
    out = tmp_path / "out"

    main([
        "batch", str(corpus), "--out-dir", str(out), "--seed", "1",
        "--set", "pitch_semitones=4", "--threshold", "privacy.max_cosine_similarity=2", "--no-content",
    ])

    report = json.loads((out / "bob" / "take0_anon.json").read_text())
    assert report["params"]["pitch_semitones"] == 4.0
    assert report["axes"]["privacy"]["verdict"] is True
    assert "content" not in report["axes"]
    assert _summary(out)["files"][0]["verdict"] in (True, False)

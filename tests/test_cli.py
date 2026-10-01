import json

from anodio.cli.main import main
from anodio.core import load_wav, save_wav


def test_anonymize_command_writes_wav_and_report(tmp_path, speech):
    source = tmp_path / "interview.wav"
    save_wav(source, speech)
    out = tmp_path / "out"

    exit_code = main(["anonymize", str(source), "--seed", "3", "--out-dir", str(out)])

    assert exit_code == 0
    anonymized = load_wav(out / "interview_anon.wav")
    assert len(anonymized.samples) == len(speech.samples)
    report = json.loads((out / "interview_anon.json").read_text())
    assert report["params"]["seed"] == 3
    assert "prosody" in report["axes"]


def test_unreadable_input_fails_with_a_message(tmp_path, capsys):
    source = tmp_path / "broken.wav"
    source.write_bytes(b"garbage")

    exit_code = main(["anonymize", str(source), "--seed", "1"])

    assert exit_code != 0
    assert "broken.wav" in capsys.readouterr().err


def test_overrides_and_stages_are_applied(tmp_path, speech):
    source = tmp_path / "a.wav"
    save_wav(source, speech)

    exit_code = main([
        "anonymize", str(source), "--seed", "1", "--out-dir", str(tmp_path),
        "--set", "pitch_semitones=4", "--stages", "mcadams,psola",
    ])

    assert exit_code == 0
    params = json.loads((tmp_path / "a_anon.json").read_text())["params"]
    assert params["pitch_semitones"] == 4.0
    assert params["stages"] == ["mcadams", "psola"]


def test_out_of_safe_range_override_warns(tmp_path, speech, capsys):
    source = tmp_path / "a.wav"
    save_wav(source, speech)

    main(["anonymize", str(source), "--seed", "1", "--out-dir", str(tmp_path), "--set", "pitch_semitones=0.5"])

    assert "pitch_semitones" in capsys.readouterr().err

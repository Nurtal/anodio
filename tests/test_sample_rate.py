"""The analysis runs on a 16 kHz copy: a Report does not depend on the Original's sample rate."""

import pytest
from synth import band_limited_utterance

from anodio.cli.main import main
from anodio.core import evaluate, load_wav, save_wav


def _report(sample_rate):
    original = band_limited_utterance(sample_rate)
    other = band_limited_utterance(sample_rate, vibrato=0.03, shift=0.01)
    return evaluate(original, other)


@pytest.fixture(scope="module")
def reports():
    return _report(16000), _report(44100)


@pytest.mark.parametrize("axis", ["clinical", "prosody"])
def test_measures_do_not_depend_on_the_sample_rate(reports, axis):
    at_16k, at_44k = (report.axes[axis].measures for report in reports)

    assert at_16k.keys() == at_44k.keys()
    for name, value in at_16k.items():
        assert at_44k[name] == pytest.approx(value, rel=2e-3, abs=1e-6), name


def test_durations_are_those_of_the_recordings_not_of_their_16k_copies():
    original = band_limited_utterance(44100)
    odd = type(original)(original.samples[:-7], 44100)  # not a whole number of samples at 16 kHz

    measures = evaluate(odd, odd).axes["prosody"].measures

    assert measures["duration_original_s"] == odd.duration


def test_exported_anonymized_keeps_the_sample_rate_of_the_original(tmp_path):
    original_path = tmp_path / "a.wav"
    save_wav(original_path, band_limited_utterance(44100))

    main(["anonymize", str(original_path), "--seed", "1", "--no-privacy", "--no-content"])

    assert load_wav(tmp_path / "a_anon.wav").sample_rate == 44100

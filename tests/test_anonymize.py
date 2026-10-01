import numpy as np
import pytest
from synth import UTTERANCE_SEGMENTS, utterance

from anodio.core import anonymize, draw_params


@pytest.mark.parametrize("sample_rate", [16000, 44100])
def test_anonymized_keeps_duration_and_sample_rate(sample_rate):
    original = utterance(sample_rate)

    anonymized = anonymize(original, draw_params(0))

    assert anonymized.sample_rate == sample_rate
    assert len(anonymized.samples) == len(original.samples)


def test_same_params_give_identical_output(speech):
    params = draw_params(3)

    np.testing.assert_array_equal(anonymize(speech, params).samples, anonymize(speech, params).samples)


def test_different_seeds_give_different_outputs(speech):
    a = anonymize(speech, draw_params(1)).samples
    b = anonymize(speech, draw_params(2)).samples

    assert not np.allclose(a, b, atol=1e-3)


def test_silences_stay_silent_in_place(speech):
    anonymized = anonymize(speech, draw_params(0))
    sr = speech.sample_rate
    # Silent stretches well clear of the voiced segments' edges (LPC frames are 20 ms).
    margin = 0.05
    edges = [0.0, *[t for seg in UTTERANCE_SEGMENTS for t in seg], len(speech.samples) / sr]
    silent = [(edges[i] + margin, edges[i + 1] - margin) for i in range(0, len(edges), 2)]

    for start, end in silent:
        stretch = anonymized.samples[int(start * sr) : int(end * sr)]
        assert np.max(np.abs(stretch)) < 1e-4, (start, end)


def test_anonymized_differs_from_original_without_clipping_or_vanishing(speech):
    anonymized = anonymize(speech, draw_params(0)).samples
    original = speech.samples

    correlation = np.corrcoef(original, anonymized)[0, 1]
    assert correlation < 0.9
    assert np.max(np.abs(anonymized)) <= 1.0
    rms_ratio = np.sqrt(np.mean(anonymized**2) / np.mean(original**2))
    assert 0.3 < rms_ratio < 3.0


def _median_f0(audio):
    import parselmouth
    from parselmouth.praat import call

    pitch = parselmouth.Sound(audio.samples.astype(np.float64), audio.sample_rate).to_pitch()
    return call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")


def test_no_stage_leaves_the_original_untouched(speech):
    anonymized = anonymize(speech, draw_params(0, stages=()))

    np.testing.assert_allclose(anonymized.samples, speech.samples, atol=1e-6)


@pytest.mark.parametrize("stage", ["mcadams", "psola", "vtln"])
def test_each_stage_alone_transforms_the_signal(speech, stage):
    anonymized = anonymize(speech, draw_params(0, stages=(stage,)))

    assert np.corrcoef(speech.samples, anonymized.samples)[0, 1] < 0.95


@pytest.mark.parametrize("semitones", [-4.0, 3.0])
def test_pitch_moves_by_the_requested_semitones(semitones):
    original = utterance(vibrato_semitones=0.0)
    params = draw_params(0, stages=("psola",), overrides={"pitch_semitones": semitones, "formant_ratio": 1.0})

    ratio = _median_f0(anonymize(original, params)) / _median_f0(original)

    assert 12 * np.log2(ratio) == pytest.approx(semitones, abs=0.3)


@pytest.mark.parametrize("seed", range(5))
def test_full_chain_keeps_duration_and_silences(seed, speech):
    anonymized = anonymize(speech, draw_params(seed))

    assert len(anonymized.samples) == len(speech.samples)
    sr = speech.sample_rate
    assert np.max(np.abs(anonymized.samples[: int(0.2 * sr)])) < 1e-4
    assert np.max(np.abs(anonymized.samples[int(2.25 * sr) :])) < 1e-4

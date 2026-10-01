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

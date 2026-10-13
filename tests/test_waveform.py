"""Waveform of the Original, shown as soon as it is loaded, then of the Anonymized recording."""

import time

import numpy as np
import pytest
from fakes import ScriptedTranscriber, SpectrumEmbedder

from anodio.core import Audio, save_wav
from anodio.core.features import waveform_envelope

FIVE_MINUTES = 300


def test_short_recording_keeps_every_sample():
    audio = Audio(np.array([0.0, 0.5, -0.5, 0.25], np.float32), 4)

    times, low, high = waveform_envelope(audio, max_columns=10)

    np.testing.assert_allclose(times, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_array_equal(low, audio.samples)
    np.testing.assert_array_equal(high, audio.samples)


def test_long_recording_is_reduced_to_a_min_max_envelope_that_keeps_its_peaks():
    samples = np.zeros(100_003, np.float32)
    samples[40_000], samples[99_999] = 0.9, -0.7
    audio = Audio(samples, 16000)

    times, low, high = waveform_envelope(audio, max_columns=500)

    assert len(times) == len(low) == len(high) <= 500
    assert high.max() == np.float32(0.9)
    assert low.min() == np.float32(-0.7)
    assert times[np.argmax(high)] == pytest.approx(40_000 / 16000, abs=audio.duration / len(times))
    assert 0 <= times[0] and times[-1] < audio.duration


def _traces(view):
    """One (title, number of drawn columns) per plotted waveform."""
    return [(ax.get_title(loc="left"), len(ax.collections[0].get_paths()[0].vertices)) for ax in view.figure.axes]


def test_view_shows_the_original_alone_then_both(app, speech):
    from anodio.gui.plots import WaveformView

    view = WaveformView()
    view.show(speech)
    assert [title for title, _ in _traces(view)] == ["Original"]

    view.show(speech, speech)
    assert [title for title, _ in _traces(view)] == ["Original", "Anonymisé"]

    view.clear()
    assert view.figure.axes == []


def test_five_minutes_are_drawn_in_under_half_a_second(app):
    from anodio.gui.plots import WaveformView

    rng = np.random.default_rng(0)
    audio = Audio(rng.uniform(-1, 1, FIVE_MINUTES * 44100).astype(np.float32), 44100)
    view = WaveformView()
    view.resize(900, 200)
    view.show(audio)  # the first draw also pays for matplotlib's one-off set-up
    view.draw()

    start = time.perf_counter()
    view.show(audio, audio)
    view.draw()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5


@pytest.fixture
def window(app, monkeypatch):
    from anodio.gui import worker
    from anodio.gui.main import MainWindow

    monkeypatch.setattr(worker, "EcapaEmbedder", SpectrumEmbedder)
    monkeypatch.setattr(worker, "WhisperTranscriber", lambda size: ScriptedTranscriber(default="bonjour"))
    return MainWindow()


def test_waveform_of_the_original_is_shown_on_load_and_the_anonymized_after_the_run(window, tmp_path, speech):
    path = tmp_path / "a.wav"
    save_wav(path, speech)

    window.load_path(path)
    assert [title for title, _ in _traces(window.waveforms)] == ["Original"]

    window.run_anonymization()
    window.wait_for_worker()
    assert [title for title, _ in _traces(window.waveforms)] == ["Original", "Anonymisé"]

    window.load_path(path)
    assert [title for title, _ in _traces(window.waveforms)] == ["Original"]

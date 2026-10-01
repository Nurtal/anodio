import numpy as np
import pytest
import soundfile as sf

from anodio.core import AnodioError, load_wav, save_wav


def test_stereo_wav_is_loaded_as_mono_with_its_sample_rate(tmp_path):
    left = np.full(22050, 0.5, dtype=np.float32)
    right = np.full(22050, -0.1, dtype=np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([left, right], axis=1), 22050)

    audio = load_wav(path)

    assert audio.sample_rate == 22050
    assert audio.samples.ndim == 1
    assert audio.samples.shape == (22050,)
    np.testing.assert_allclose(audio.samples, 0.2, atol=1e-4)


def test_saved_wav_reloads_identically(tmp_path):
    from anodio.core import Audio

    audio = Audio(np.linspace(-0.5, 0.5, 44100, dtype=np.float32), 44100)
    path = tmp_path / "out.wav"
    save_wav(path, audio)

    reloaded = load_wav(path)

    assert reloaded.sample_rate == 44100
    np.testing.assert_allclose(reloaded.samples, audio.samples, atol=1e-6)


def test_unreadable_file_raises_a_clear_error(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wav file at all")

    with pytest.raises(AnodioError, match="broken.wav"):
        load_wav(path)


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(AnodioError, match="introuvable"):
        load_wav(tmp_path / "nope.wav")


def test_too_short_recording_is_rejected(tmp_path):
    path = tmp_path / "click.wav"
    sf.write(path, np.full(400, 0.1, dtype=np.float32), 16000)

    with pytest.raises(AnodioError, match="trop court"):
        load_wav(path)

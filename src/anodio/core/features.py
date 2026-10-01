"""Acoustic measures shared by the Prosody and Clinical Axes (Praat via parselmouth)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

from anodio.core.audio import Audio
from anodio.core.timing import HOP_SECONDS, SILENCE_DB, frame_rms, pauses

PITCH_FLOOR = 60.0
PITCH_CEILING = 500.0
FIRST_PASS_OCTAVE_COST = 0.05
NUCLEUS_PROMINENCE_DB = 2.0
NUCLEUS_FLOOR_DB = 25.0  # below the loudest frame
ENERGY_FLOOR_DB = -60.0
DTW_BAND_FRAMES = 10


@dataclass(frozen=True)
class F0Contour:
    times: np.ndarray
    hz: np.ndarray  # 0 where unvoiced

    @property
    def voiced(self) -> np.ndarray:
        return self.hz > 0

    def semitones_from_median(self) -> np.ndarray:
        """Contour in semitones relative to the Speaker's median F0 (NaN where unvoiced)."""
        out = np.full(len(self.hz), np.nan)
        if self.voiced.any():
            median = np.median(self.hz[self.voiced])
            out[self.voiced] = 12 * np.log2(self.hz[self.voiced] / median)
        return out


def _sound(audio: Audio) -> parselmouth.Sound:
    return parselmouth.Sound(audio.samples.astype(np.float64), audio.sample_rate)


def pitch_range(sound: parselmouth.Sound) -> tuple[float, float]:
    """Speaker-adapted pitch range (Hirst's two-pass method) to avoid octave errors.

    The first pass raises the octave cost so that sub-harmonic candidates do not drag
    the 15th percentile, and thus the floor, an octave down.
    """
    first = sound.to_pitch_ac(
        time_step=HOP_SECONDS,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING,
        octave_cost=FIRST_PASS_OCTAVE_COST,
    )
    hz = np.asarray(first.selected_array["frequency"])
    hz = hz[hz > 0]
    if len(hz) < 10:
        return PITCH_FLOOR, PITCH_CEILING
    q15, q85 = np.quantile(hz, [0.15, 0.85])
    return max(0.75 * q15, PITCH_FLOOR / 2), min(1.5 * q85, PITCH_CEILING * 1.5)


def f0_contour(audio: Audio) -> F0Contour:
    sound = _sound(audio)
    floor, ceiling = pitch_range(sound)
    pitch = sound.to_pitch_ac(time_step=HOP_SECONDS, pitch_floor=floor, pitch_ceiling=ceiling)
    return F0Contour(np.asarray(pitch.xs()), np.asarray(pitch.selected_array["frequency"]))


def f0_correlation(a: F0Contour, b: F0Contour) -> float | None:
    """Pearson correlation of the normalized contours over frames voiced in both.

    `b` is resampled (nearest frame) on the time grid of `a`: the analysis window,
    hence the frame times, depends on each signal's pitch range.
    """
    if len(a.hz) == 0 or len(b.hz) == 0:
        return None
    nearest = np.clip(np.searchsorted(b.times, a.times), 0, len(b.times) - 1)
    previous = np.clip(nearest - 1, 0, len(b.times) - 1)
    nearest = np.where(np.abs(b.times[previous] - a.times) < np.abs(b.times[nearest] - a.times), previous, nearest)
    sa, sb = a.semitones_from_median(), b.semitones_from_median()[nearest]
    both = ~np.isnan(sa) & ~np.isnan(sb)
    if both.sum() < 3:
        return None
    sa, sb = sa[both], sb[both]
    if sa.std() < 1e-6 or sb.std() < 1e-6:
        return 1.0 if sa.std() < 1e-6 and sb.std() < 1e-6 else 0.0
    return float(np.corrcoef(sa, sb)[0, 1])


def energy_db(audio: Audio) -> np.ndarray:
    rms = frame_rms(audio.samples, audio.sample_rate)
    peak = max(float(rms.max()), 1e-12)
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-12) / peak), ENERGY_FLOOR_DB)


def energy_dtw_db(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND_FRAMES) -> float:
    """Mean absolute dB difference along the best DTW path, within a Sakoe-Chiba band."""
    n, m = len(a), len(b)
    band = max(band, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            prev = min((cost[i - 1, j - 1], i - 1, j - 1), (cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1))
            cost[i, j] = prev[0] + abs(a[i - 1] - b[j - 1])
            steps[i, j] = steps[prev[1], prev[2]] + 1
    return float(cost[n, m] / steps[n, m])


def clinical_features(audio: Audio, contour: F0Contour | None = None) -> dict[str, float]:
    """Relative or normalized voice-quality and rate measures. Never an absolute F0."""
    contour = contour or f0_contour(audio)
    voiced_hz = contour.hz[contour.voiced]
    sound = _sound(audio)
    floor, ceiling = pitch_range(sound)
    point_process = call(sound, "To PointProcess (periodic, cc)", floor, ceiling)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", HOP_SECONDS, floor, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pause_list = pauses(audio.samples, audio.sample_rate)
    semitones = contour.semitones_from_median()
    return {
        "f0_cv": float(voiced_hz.std() / voiced_hz.mean()) if len(voiced_hz) else float("nan"),
        "f0_sd_semitones": float(np.nanstd(semitones)) if len(voiced_hz) else float("nan"),
        "jitter_local_pct": 100 * float(jitter),
        "shimmer_local_pct": 100 * float(shimmer),
        "hnr_db": float(hnr),
        "speech_rate_syll_per_s": _speech_rate(sound, contour, audio, pause_list),
        "pause_count": float(len(pause_list)),
        "mean_pause_s": float(np.mean([e - s for s, e in pause_list])) if pause_list else 0.0,
    }


def _speech_rate(sound: parselmouth.Sound, contour: F0Contour, audio: Audio, pause_list) -> float:
    """Syllable nuclei (voiced intensity peaks) per second of speaking time."""
    intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=HOP_SECONDS)
    db = np.asarray(intensity.values[0])
    times = np.asarray(intensity.xs())
    peaks, _ = find_peaks(db, prominence=NUCLEUS_PROMINENCE_DB, height=db.max() - NUCLEUS_FLOOR_DB)
    voiced_at = np.interp(times[peaks], contour.times, contour.voiced.astype(float)) > 0.5 if len(peaks) else []
    nuclei = int(np.sum(voiced_at))
    active = energy_db(audio) > SILENCE_DB
    if not active.any():
        return 0.0
    first, last = np.flatnonzero(active)[[0, -1]]
    speaking = (last - first + 1) * HOP_SECONDS - sum(e - s for s, e in pause_list)
    return nuclei / speaking if speaking > 0 else 0.0


MFCC_COUNT = 13
MEL_FILTERS = 26


def mfcc(audio: Audio) -> np.ndarray:
    """MFCC c1..c13 (frames x 13) of the non-silent 25 ms frames."""
    from scipy.fft import dct

    sr = audio.sample_rate
    x = np.append(audio.samples[0], audio.samples[1:] - 0.97 * audio.samples[:-1]).astype(np.float64)
    frame, hop = int(0.025 * sr), int(HOP_SECONDS * sr)
    if len(x) < frame:
        return np.zeros((0, MFCC_COUNT))
    n_fft = 1 << (frame - 1).bit_length()
    starts = np.arange(0, len(x) - frame + 1, hop)
    frames = np.stack([x[s : s + frame] for s in starts]) * np.hamming(frame)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    log_mel = np.log(power @ _mel_filterbank(sr, n_fft).T + 1e-10)
    coefficients = dct(log_mel, type=2, norm="ortho", axis=1)[:, 1 : MFCC_COUNT + 1]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    speech = 20 * np.log10(np.maximum(rms, 1e-12) / max(rms.max(), 1e-12)) > SILENCE_DB
    return coefficients[speech]


def _mel_filterbank(sample_rate: int, n_fft: int) -> np.ndarray:
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    edges = to_hz(np.linspace(0, to_mel(sample_rate / 2), MEL_FILTERS + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    bank = np.zeros((MEL_FILTERS, len(bins)))
    for i in range(MEL_FILTERS):
        low, centre, high = edges[i : i + 3]
        rising = (bins - low) / (centre - low)
        falling = (high - bins) / (high - centre)
        bank[i] = np.maximum(0, np.minimum(rising, falling))
    return bank

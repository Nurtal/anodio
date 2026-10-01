import dataclasses

import pytest

from anodio.core import IDENTITY, SAFE_RANGES, STAGES, draw_params

PARAM_NAMES = ["mcadams_alpha", "pitch_semitones", "formant_ratio", "vtln_alpha"]


def _in_safe_range(name, value):
    return any(low <= value <= high for low, high in SAFE_RANGES[name])


@pytest.mark.parametrize("seed", range(50))
def test_drawn_params_stay_in_safe_ranges(seed):
    params = draw_params(seed)
    for name in PARAM_NAMES:
        assert _in_safe_range(name, getattr(params, name)), name


@pytest.mark.parametrize("seed", range(50))
def test_drawn_params_avoid_the_identity(seed):
    params = draw_params(seed)
    assert abs(params.pitch_semitones) >= 2
    assert abs(params.formant_ratio - 1) >= 0.05
    assert abs(params.vtln_alpha) >= 0.05
    assert params.mcadams_alpha <= 0.9


def test_both_directions_are_drawn():
    signs = {draw_params(seed).pitch_semitones > 0 for seed in range(50)}
    assert signs == {True, False}


def test_same_seed_gives_same_params():
    assert draw_params(42) == draw_params(42)


def test_different_seeds_give_different_params():
    assert draw_params(1) != draw_params(2)


def test_params_record_their_seed():
    assert draw_params(7).seed == 7


def test_override_replaces_only_its_param():
    drawn = draw_params(9)

    overridden = draw_params(9, overrides={"pitch_semitones": 3.5})

    assert overridden.pitch_semitones == 3.5
    assert dataclasses.replace(overridden, pitch_semitones=drawn.pitch_semitones) == drawn


def test_all_stages_enabled_by_default():
    assert draw_params(0).stages == STAGES


def test_stages_can_be_selected():
    assert draw_params(0, stages=("psola",)).stages == ("psola",)


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError):
        draw_params(0, overrides={"nope": 1.0})


def test_out_of_safe_range_override_is_flagged():
    params = draw_params(0, overrides={"pitch_semitones": 0.5, "formant_ratio": 1.1})

    assert params.outside_safe_ranges() == ["pitch_semitones"]


def test_identity_params_are_outside_safe_ranges():
    assert set(IDENTITY) == set(PARAM_NAMES)
    params = draw_params(0, overrides=IDENTITY)
    assert params.outside_safe_ranges() == PARAM_NAMES

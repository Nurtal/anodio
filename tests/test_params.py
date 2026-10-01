import pytest

from anodio.core import SAFE_RANGES, draw_params


@pytest.mark.parametrize("seed", range(50))
def test_drawn_params_stay_in_safe_ranges(seed):
    params = draw_params(seed)
    low, high = SAFE_RANGES["mcadams_alpha"]
    assert low <= params.mcadams_alpha <= high


def test_same_seed_gives_same_params():
    assert draw_params(42) == draw_params(42)


def test_different_seeds_give_different_params():
    assert draw_params(1) != draw_params(2)


def test_params_record_their_seed():
    assert draw_params(7).seed == 7

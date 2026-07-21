import math

import pytest

from avlnn.atmosphere import isa
from avlnn.config import Constants


@pytest.fixture
def atmo():
    return Constants.load().atmosphere


def test_sea_level(atmo):
    state = isa(0.0, atmo)
    assert state.temperature_K == pytest.approx(288.15, abs=0.01)
    assert state.pressure_Pa == pytest.approx(101325.0, abs=1.0)
    assert state.density_kg_m3 == pytest.approx(1.225, abs=0.001)


def test_tropopause_11km_matches_isa_table(atmo):
    """Reference values from the standard ISA table at 11,000 m (the cruise altitude)."""
    state = isa(11000.0, atmo)
    assert state.temperature_K == pytest.approx(216.65, abs=0.05)
    assert state.pressure_Pa == pytest.approx(22632.0, rel=0.01)
    assert state.density_kg_m3 == pytest.approx(0.3639, rel=0.01)
    assert state.speed_of_sound_m_s == pytest.approx(295.07, rel=0.01)


def test_negative_altitude_rejected(atmo):
    with pytest.raises(ValueError):
        isa(-100.0, atmo)


def test_isothermal_layer_above_tropopause_continuous(atmo):
    just_below = isa(10999.0, atmo)
    just_above = isa(11001.0, atmo)
    assert just_below.temperature_K == pytest.approx(just_above.temperature_K, abs=0.01)
    assert just_below.pressure_Pa == pytest.approx(just_above.pressure_Pa, rel=1e-3)

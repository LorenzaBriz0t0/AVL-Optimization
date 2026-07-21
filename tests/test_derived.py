import math

import pytest

from avlnn.config import Constants
from avlnn.derived import (
    aircraft_cg_x_b,
    assemble,
    available_thrust_N,
    cruise_cl,
    drag_coefficient,
    oswald_efficiency,
    wing_geometry,
)
from avlnn.design_space import DesignSpace


@pytest.fixture
def constants():
    return Constants.load()


@pytest.fixture
def space():
    return DesignSpace.load()


@pytest.fixture
def baseline_design(space):
    x = (space.lower + space.upper) / 2.0
    return space.to_dict(x)


def test_wing_geometry_span_and_area(baseline_design):
    wing = wing_geometry(baseline_design)
    S = baseline_design["wing_area_m2"]
    AR = baseline_design["wing_aspect_ratio"]
    assert wing.span_m == pytest.approx(math.sqrt(S * AR))
    # Trapezoidal area check: S = b/2 * (c_root + c_tip)
    area_check = wing.span_m / 2.0 * (wing.root_chord_m + wing.tip_chord_m)
    assert area_check == pytest.approx(S, rel=1e-6)


def test_wing_geometry_rectangular_mac_equals_chord():
    """For taper ratio 1 (rectangular wing), the MAC must equal the constant chord."""
    design = {
        "wing_area_m2": 15.0, "wing_aspect_ratio": 6.0, "wing_taper_ratio": 1.0,
        "wing_sweep_c4_deg": 0.0, "wing_dihedral_deg": 0.0,
    }
    wing = wing_geometry(design)
    expected_chord = design["wing_area_m2"] / wing.span_m
    assert wing.root_chord_m == pytest.approx(expected_chord)
    assert wing.tip_chord_m == pytest.approx(expected_chord)
    assert wing.mean_aero_chord_m == pytest.approx(expected_chord)


def test_oswald_efficiency_reasonable_range():
    for ar in (4.0, 6.0, 8.0, 10.0):
        e = oswald_efficiency(ar)
        assert 0.5 < e < 1.0


def test_cruise_cl_and_drag_coefficient():
    cl = cruise_cl(weight_N=40000.0, dynamic_pressure_Pa=10000.0, wing_area_m2=15.0)
    assert cl == pytest.approx(40000.0 / (10000.0 * 15.0))
    cd = drag_coefficient(cl, aspect_ratio=8.0, cd0=0.03)
    assert cd > 0.03  # induced drag must add to CD0, not subtract


def test_available_thrust_zero_at_mach_1():
    t = available_thrust_N(
        static_thrust_to_weight_sl=1.5, weight_N=40000.0, mach=1.0,
        thrust_zero_mach=1.0, rho_at_altitude=1.225, rho_sea_level=1.225,
    )
    assert t == pytest.approx(0.0, abs=1e-6)


def test_available_thrust_static_at_mach_0_sea_level():
    t = available_thrust_N(
        static_thrust_to_weight_sl=1.5, weight_N=40000.0, mach=0.0,
        thrust_zero_mach=1.0, rho_at_altitude=1.225, rho_sea_level=1.225,
    )
    assert t == pytest.approx(1.5 * 40000.0)


def test_aircraft_cg_symmetric_fuselage_contributes_zero(baseline_design, constants):
    wing = wing_geometry(baseline_design)
    x_cg = aircraft_cg_x_b(baseline_design, wing, constants)
    wing_mass_point_x_b = baseline_design["wing_root_le_x_b_m"] + wing.mass_point_x_b_m
    expected = 2.0 * constants.mass.wing_fraction_each * wing_mass_point_x_b
    assert x_cg == pytest.approx(expected)


def test_assemble_mass_fractions_sum_to_one(constants):
    total_frac = constants.mass.fuselage_fraction + 2 * constants.mass.wing_fraction_each
    assert total_frac == pytest.approx(1.0)


def test_assemble_produces_consistent_geometry(baseline_design, constants):
    geom = assemble(baseline_design, constants)
    assert geom.sref_m2 == pytest.approx(baseline_design["wing_area_m2"])
    assert geom.htail.area_m2 > 0
    assert geom.vtail.area_m2 > 0
    assert geom.htail.ac_x_b_m == pytest.approx(geom.x_cg_b_m + baseline_design["htail_arm_m"])

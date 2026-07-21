import pytest

from avlnn.config import Constants
from avlnn.derived import assemble
from avlnn.design_space import DesignSpace
from avlnn.massfile import build_avl_mass_file


@pytest.fixture
def constants():
    return Constants.load()


@pytest.fixture
def baseline_design():
    space = DesignSpace.load()
    x = (space.lower + space.upper) / 2.0
    return space.to_dict(x)


def test_mass_file_totals_4000kg(baseline_design, constants):
    geom = assemble(baseline_design, constants)
    text = build_avl_mass_file(baseline_design, geom, constants)
    data_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#") and "=" not in l
    ]
    total = sum(float(l.split()[0]) for l in data_lines)
    assert total == pytest.approx(constants.mass.total_kg, rel=1e-6)


def test_mass_file_has_22_point_masses(baseline_design, constants):
    geom = assemble(baseline_design, constants)
    text = build_avl_mass_file(baseline_design, geom, constants)
    data_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#") and "=" not in l
    ]
    # 20 fuselage points (5 rows x 4) + 2 wing points
    assert len(data_lines) == 22


def test_mass_file_symmetric_about_y_and_z(baseline_design, constants):
    geom = assemble(baseline_design, constants)
    text = build_avl_mass_file(baseline_design, geom, constants)
    data_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#") and "=" not in l
    ]
    y_moment = sum(float(l.split()[0]) * float(l.split()[2]) for l in data_lines)
    z_moment = sum(float(l.split()[0]) * float(l.split()[3]) for l in data_lines)
    assert y_moment == pytest.approx(0.0, abs=1e-6)
    assert z_moment == pytest.approx(0.0, abs=1e-6)


def test_mass_file_x_cg_matches_derived_cg(baseline_design, constants):
    geom = assemble(baseline_design, constants)
    text = build_avl_mass_file(baseline_design, geom, constants)
    data_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#") and "=" not in l
    ]
    total_mass = sum(float(l.split()[0]) for l in data_lines)
    x_moment = sum(float(l.split()[0]) * float(l.split()[1]) for l in data_lines)
    x_cg_from_points = x_moment / total_mass
    # The mass file writes coordinates with 5 decimal places, so the CG recomputed from the
    # file text is quantized to ~1e-5 m; a relative tolerance would be far tighter than the
    # format allows whenever the CG sits near x_b = 0.
    assert x_cg_from_points == pytest.approx(geom.x_cg_b_m, abs=1e-5)

import pytest

from avlnn.config import Constants
from avlnn.design_space import DesignSpace
from avlnn.geometry import build_avl_geometry


@pytest.fixture
def constants():
    return Constants.load()


@pytest.fixture
def baseline_design():
    space = DesignSpace.load()
    x = (space.lower + space.upper) / 2.0
    return space.to_dict(x)


def test_geometry_contains_required_surfaces(baseline_design, constants):
    text, geom = build_avl_geometry(baseline_design, constants)
    assert "SURFACE" in text
    assert "Wing" in text
    assert "Horizontal Tail" in text
    assert "Vertical Tail" in text
    assert text.count("SURFACE") == 3


def test_geometry_uses_naca_2312_on_wing_only(baseline_design, constants):
    text, _geom = build_avl_geometry(baseline_design, constants)
    assert "2312" in text
    # tails must NOT carry an airfoil card -- an airfoil-less SECTION is the flat-plate model
    wing_block = text.split("Horizontal Tail")[0]
    tail_blocks = text.split("Horizontal Tail")[1]
    assert "NACA" in wing_block
    assert "NACA" not in tail_blocks


def test_geometry_header_matches_wing_reference_values(baseline_design, constants):
    text, geom = build_avl_geometry(baseline_design, constants)
    lines = text.splitlines()
    sref_line_idx = next(i for i, l in enumerate(lines) if l.startswith("#Sref"))
    sref, cref, bref = (float(v) for v in lines[sref_line_idx + 1].split())
    assert sref == pytest.approx(geom.sref_m2, rel=1e-4)
    assert cref == pytest.approx(geom.cref_m, rel=1e-4)
    assert bref == pytest.approx(geom.bref_m, rel=1e-4)


def test_geometry_control_surfaces_present(baseline_design, constants):
    text, _geom = build_avl_geometry(baseline_design, constants)
    assert "aileron" in text
    assert "elevator" in text
    assert "rudder" in text


def test_geometry_has_control_surface_count_matching_wing_sections(baseline_design, constants):
    text, _geom = build_avl_geometry(baseline_design, constants)
    wing_block = text.split("Horizontal Tail")[0]
    # aileron CONTROL card appears on the 2 outboard wing sections (mid-span + tip)
    assert wing_block.count("aileron") == 2

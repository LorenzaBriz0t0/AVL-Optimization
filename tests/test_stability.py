import pytest

from avlnn.config import Constants
from avlnn.parse import Eigenvalue
from avlnn.stability import check_dynamic_modes, classify_modes, evaluate_stability, static_margin


@pytest.fixture
def constants():
    return Constants.load()


def _stable_eigenvalue_set() -> list[Eigenvalue]:
    """2 real roots (roll, spiral) + 3 complex-conjugate pairs (phugoid, dutch roll, short
    period), all stable, in a plausible frequency ordering for a trainer aircraft."""
    return [
        Eigenvalue(-8.0, 0.0),    # roll (fast, real)
        Eigenvalue(-0.02, 0.0),   # spiral (slow, real, stable)
        Eigenvalue(-0.05, 0.3), Eigenvalue(-0.05, -0.3),   # phugoid (low freq)
        Eigenvalue(-0.3, 1.5), Eigenvalue(-0.3, -1.5),     # dutch roll (mid freq)
        Eigenvalue(-2.0, 4.0), Eigenvalue(-2.0, -4.0),     # short period (high freq)
    ]


def test_classify_modes_assigns_expected_roles():
    modes = classify_modes(_stable_eigenvalue_set())
    assert modes["roll"][0].real == pytest.approx(-8.0)
    assert modes["spiral"][0].real == pytest.approx(-0.02)
    assert modes["phugoid"][0].frequency_rad_s == pytest.approx(0.3)
    assert modes["dutch_roll"][0].frequency_rad_s == pytest.approx(1.5)
    assert modes["short_period"][0].frequency_rad_s == pytest.approx(4.0)
    assert len(modes["phugoid"]) == 2  # conjugate pair found


def test_classify_modes_too_few_real_roots_raises():
    with pytest.raises(ValueError):
        classify_modes([Eigenvalue(-1.0, 2.0), Eigenvalue(-1.0, -2.0)])


def test_check_dynamic_modes_all_stable_passes(constants):
    checks = check_dynamic_modes(_stable_eigenvalue_set(), constants)
    assert all(c.passed for c in checks.values())


def test_spiral_unstable_but_slow_enough_passes(constants):
    eigs = _stable_eigenvalue_set()
    eigs[1] = Eigenvalue(1.0 / 25.0, 0.0)  # tau = 25s >= 20s minimum -> allowed
    checks = check_dynamic_modes(eigs, constants)
    assert checks["spiral"].passed


def test_spiral_unstable_too_fast_fails(constants):
    eigs = _stable_eigenvalue_set()
    eigs[1] = Eigenvalue(1.0 / 10.0, 0.0)  # tau = 10s < 20s minimum -> not allowed
    checks = check_dynamic_modes(eigs, constants)
    assert not checks["spiral"].passed


def test_non_spiral_mode_unstable_fails(constants):
    eigs = _stable_eigenvalue_set()
    eigs[6] = Eigenvalue(0.1, 4.0)   # short period made unstable
    eigs[7] = Eigenvalue(0.1, -4.0)
    checks = check_dynamic_modes(eigs, constants)
    assert not checks["short_period"].passed


def test_static_margin_sign():
    assert static_margin(x_np=1.0, x_cg=0.5, cref=1.0) > 0    # CG ahead of NP -> stable
    assert static_margin(x_np=0.5, x_cg=1.0, cref=1.0) < 0    # CG behind NP -> unstable


def test_evaluate_stability_all_passed(constants):
    report = evaluate_stability(
        x_np=1.0, x_cg=0.5, cref=1.0, alpha_trim_deg=5.0, wing_incidence_deg=0.0,
        eigenvalues=_stable_eigenvalue_set(), c=constants,
    )
    assert report.static_margin_passed
    assert report.stall_margin_passed
    assert report.all_passed


def test_evaluate_stability_stall_margin_fails_above_limit(constants):
    report = evaluate_stability(
        x_np=1.0, x_cg=0.5, cref=1.0, alpha_trim_deg=12.0,  # exceeds 10 deg stall limit
        wing_incidence_deg=0.0,
        eigenvalues=_stable_eigenvalue_set(), c=constants,
    )
    assert not report.stall_margin_passed
    assert not report.all_passed


def test_evaluate_stability_incidence_counts_toward_stall(constants):
    """The 10 deg limit applies to the wing section's AoA (trim alpha + incidence), so a
    trim alpha that passes alone must fail once incidence pushes the sum over the limit."""
    report = evaluate_stability(
        x_np=1.0, x_cg=0.5, cref=1.0, alpha_trim_deg=7.0, wing_incidence_deg=4.0,
        eigenvalues=_stable_eigenvalue_set(), c=constants,
    )
    assert not report.stall_margin_passed
    assert report.stall_margin_deg == pytest.approx(10.0 - 11.0)

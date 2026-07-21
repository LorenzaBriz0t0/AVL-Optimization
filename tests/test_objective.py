import pytest

from avlnn.config import Constants
from avlnn.objective import TIEBREAK_WEIGHT, evaluate_objective, infeasible_result
from avlnn.parse import Eigenvalue, StabilityDerivatives
from avlnn.stability import ModeCheck, StabilityReport


@pytest.fixture
def constants():
    return Constants.load()


def _report(
    static_margin=0.30, stall_margin=5.0, mode_margin_scale=1.0, all_pass=True,
) -> StabilityReport:
    margins = {
        "phugoid": 0.02, "short_period": 2.0, "dutch_roll": 0.5, "roll": 5.0, "spiral": 50.0,
    }
    mode_checks = {
        name: ModeCheck(
            name=name, eigenvalues=(Eigenvalue(-m * mode_margin_scale, 0.0),),
            passed=all_pass, margin=m * mode_margin_scale,
        )
        for name, m in margins.items()
    }
    return StabilityReport(
        static_margin=static_margin, static_margin_passed=static_margin > 0,
        mode_checks=mode_checks,
        stall_margin_deg=stall_margin, stall_margin_passed=stall_margin > 0,
    )


def _deriv(cl=0.5) -> StabilityDerivatives:
    return StabilityDerivatives(
        cl=cl, cd=0.04, cm=0.0, alpha_deg=3.0, cl_alpha=5.0, cm_alpha=-1.0, x_np=0.5,
    )


DESIGN = {"wing_aspect_ratio": 8.0}


def test_feasible_design_gets_bounded_bonus(constants):
    result = evaluate_objective(_deriv(), _report(), DESIGN, wing_area_m2=15.0, c=constants)
    bonus = result.fitness - result.l_over_d
    assert 0.0 < bonus <= TIEBREAK_WEIGHT


def test_larger_margins_win_the_tiebreak(constants):
    tight = evaluate_objective(
        _deriv(), _report(static_margin=0.02, stall_margin=0.5, mode_margin_scale=0.1),
        DESIGN, wing_area_m2=15.0, c=constants,
    )
    comfy = evaluate_objective(
        _deriv(), _report(), DESIGN, wing_area_m2=15.0, c=constants,
    )
    assert tight.l_over_d == comfy.l_over_d          # identical aero
    assert comfy.fitness > tight.fitness             # margins break the tie


def test_failed_constraint_gets_penalty_and_no_bonus(constants):
    failing = _report(static_margin=-0.1)
    result = evaluate_objective(_deriv(), failing, DESIGN, wing_area_m2=15.0, c=constants)
    assert result.fitness < result.l_over_d          # penalized, no bonus


def test_infeasible_result_fitness_floor():
    result = infeasible_result("AVL exploded")
    assert result.fitness == -1000.0
    assert result.stability is None
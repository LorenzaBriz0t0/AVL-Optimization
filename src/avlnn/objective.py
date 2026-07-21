"""Combines AVL-derived stability/performance data into a single scalar fitness for the EA.

Fitness = cruise L/D, minus penalties proportional to how far each hard constraint (static
margin, each dynamic mode, stall margin, thrust-vs-drag feasibility) is violated. Proportional
(not flat) penalties give the EA a gradient to climb even from infeasible starting genomes.

Fully feasible designs additionally get a small saturating stability-margin bonus (capped at
TIEBREAK_WEIGHT, i.e. well under 1% of a typical L/D). Rationale: in this problem max L/D
depends only on aspect ratio and wing area, so the many remaining design variables (sweep,
dihedral, tails, ...) affect fitness solely through the constraints -- without a tie-break the
EA has no reason to prefer a robustly stable design over one that scrapes past every check.
The tanh saturation keeps huge margins (e.g. a spiral time constant of hundreds of seconds)
from buying more than the cap, so the bonus orders equal-L/D designs without ever trading
meaningfully against L/D itself.
"""
from __future__ import annotations

import dataclasses
import math

from avlnn.atmosphere import isa
from avlnn.config import Constants
from avlnn.derived import available_thrust_N, drag_coefficient
from avlnn.parse import StabilityDerivatives
from avlnn.stability import StabilityReport

INFEASIBLE_FITNESS = -1000.0
PENALTY_WEIGHT = 100.0
TIEBREAK_WEIGHT = 0.05

# Characteristic "comfortable margin" scales: tanh(margin/scale) is ~0.76 at the scale value
# and saturates toward 1 beyond ~2x it. Units match each check's margin (static margin in
# fractions of cref, stall in degrees, modes in 1/s damping -- spiral in seconds of tau slack).
_MARGIN_SCALES = {
    "static": 0.10,
    "stall_deg": 3.0,
    "phugoid": 0.05,
    "short_period": 2.0,
    "dutch_roll": 0.5,
    "roll": 3.0,
    "spiral": 60.0,
}


def _stability_margin_bonus(stability: StabilityReport) -> float:
    terms = [
        math.tanh(stability.static_margin / _MARGIN_SCALES["static"]),
        math.tanh(stability.stall_margin_deg / _MARGIN_SCALES["stall_deg"]),
    ]
    for name, check in stability.mode_checks.items():
        terms.append(math.tanh(check.margin / _MARGIN_SCALES[name]))
    return TIEBREAK_WEIGHT * sum(terms) / len(terms)


@dataclasses.dataclass(frozen=True)
class ObjectiveResult:
    fitness: float
    l_over_d: float
    thrust_margin_N: float
    stability: StabilityReport | None
    infeasible_reason: str | None = None


def evaluate_objective(
    deriv: StabilityDerivatives,
    stability: StabilityReport,
    design: dict[str, float],
    wing_area_m2: float,
    c: Constants,
) -> ObjectiveResult:
    atm = isa(c.mission.altitude_cruise_m, c.atmosphere)
    velocity = c.mission.mach_cruise * atm.speed_of_sound_m_s
    q = 0.5 * atm.density_kg_m3 * velocity**2
    weight_N = c.mass.total_kg * c.atmosphere.g0_m_s2

    cd = drag_coefficient(deriv.cl, design["wing_aspect_ratio"], c.aero.cd0)
    l_over_d = deriv.cl / cd if cd > 0 else 0.0

    drag_N = cd * q * wing_area_m2
    thrust_avail_N = available_thrust_N(
        c.propulsion.static_thrust_to_weight_sl, weight_N, c.mission.mach_cruise,
        c.propulsion.thrust_zero_mach, atm.density_kg_m3, c.atmosphere.rho0_sea_level_kg_m3,
    )
    thrust_margin_N = thrust_avail_N - drag_N

    penalty = 0.0
    if not stability.static_margin_passed:
        penalty += PENALTY_WEIGHT * abs(stability.static_margin)
    if not stability.stall_margin_passed:
        penalty += PENALTY_WEIGHT * abs(stability.stall_margin_deg)
    for mode in stability.mode_checks.values():
        if not mode.passed:
            penalty += PENALTY_WEIGHT * abs(mode.margin)
    if thrust_margin_N < 0:
        penalty += PENALTY_WEIGHT * abs(thrust_margin_N) / weight_N

    # Bonus only for fully feasible designs; infeasible ones follow the pure penalty
    # gradient toward feasibility first.
    bonus = _stability_margin_bonus(stability) if penalty == 0.0 else 0.0

    return ObjectiveResult(
        fitness=l_over_d + bonus - penalty, l_over_d=l_over_d,
        thrust_margin_N=thrust_margin_N, stability=stability,
    )


def infeasible_result(reason: str) -> ObjectiveResult:
    """Used when AVL fails to run/converge/parse -- keeps the EA/dataset gen from crashing."""
    return ObjectiveResult(
        fitness=INFEASIBLE_FITNESS, l_over_d=0.0, thrust_margin_N=float("-inf"),
        stability=None, infeasible_reason=reason,
    )

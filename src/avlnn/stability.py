"""Turns parsed AVL output into the assignment's pass/fail stability requirements.

Mode classification heuristic (AVL's eigenvalue dump does not label modes by name): the two
real roots are split into roll (larger |real|, i.e. faster) and spiral (smaller |real|); the
remaining complex-conjugate pairs are sorted by frequency and assigned, lowest to highest, to
phugoid, dutch roll, and short period. This matches typical trainer-aircraft mode ordering but
is a simplification -- sanity-check it against the eigenvectors AVL prints once real MODE
output is available (see parse.py's module docstring).
"""
from __future__ import annotations

import dataclasses

from avlnn.config import Constants
from avlnn.parse import Eigenvalue

MODE_NAMES = ("phugoid", "short_period", "dutch_roll", "roll", "spiral")


@dataclasses.dataclass(frozen=True)
class ModeCheck:
    name: str
    eigenvalues: tuple[Eigenvalue, ...]
    passed: bool
    margin: float   # for stable modes: -max(real) (bigger=more stable); for spiral: tau - tau_min


@dataclasses.dataclass(frozen=True)
class StabilityReport:
    static_margin: float
    static_margin_passed: bool
    mode_checks: dict[str, ModeCheck]
    stall_margin_deg: float
    stall_margin_passed: bool

    @property
    def all_passed(self) -> bool:
        return (
            self.static_margin_passed
            and self.stall_margin_passed
            and all(m.passed for m in self.mode_checks.values())
        )


def static_margin(x_np: float, x_cg: float, cref: float) -> float:
    """(Xnp - Xcg)/cref; positive means CG is ahead of the neutral point (stable)."""
    return (x_np - x_cg) / cref


def classify_modes(eigenvalues: list[Eigenvalue]) -> dict[str, tuple[Eigenvalue, ...]]:
    real_roots = sorted((e for e in eigenvalues if e.is_real_root), key=lambda e: e.real)
    complex_reps = sorted(
        (e for e in eigenvalues if not e.is_real_root and e.imag > 0),
        key=lambda e: e.frequency_rad_s,
    )

    if len(real_roots) < 2:
        raise ValueError(f"expected >=2 real eigenvalues (roll, spiral), got {len(real_roots)}")
    if len(complex_reps) < 2:
        raise ValueError(
            f"expected >=2 complex-pair modes (phugoid, short period [, dutch roll]), "
            f"got {len(complex_reps)}"
        )

    roll = min(real_roots, key=lambda e: e.real)      # most negative = fastest = roll subsidence
    spiral = max(real_roots, key=lambda e: e.real)     # closest to zero (or positive) = spiral

    phugoid = complex_reps[0]
    short_period = complex_reps[-1]
    dutch_roll = complex_reps[1] if len(complex_reps) >= 3 else complex_reps[len(complex_reps) // 2]

    def pair(rep: Eigenvalue) -> tuple[Eigenvalue, ...]:
        conj = next(
            (e for e in eigenvalues if abs(e.real - rep.real) < 1e-9 and abs(e.imag + rep.imag) < 1e-9),
            None,
        )
        return (rep, conj) if conj is not None else (rep,)

    return {
        "phugoid": pair(phugoid),
        "short_period": pair(short_period),
        "dutch_roll": pair(dutch_roll),
        "roll": (roll,),
        "spiral": (spiral,),
    }


def check_dynamic_modes(
    eigenvalues: list[Eigenvalue], c: Constants,
) -> dict[str, ModeCheck]:
    modes = classify_modes(eigenvalues)
    checks: dict[str, ModeCheck] = {}

    for name in ("phugoid", "short_period", "dutch_roll", "roll"):
        eigs = modes[name]
        worst_real = max(e.real for e in eigs)
        checks[name] = ModeCheck(
            name=name, eigenvalues=eigs, passed=worst_real < 0.0, margin=-worst_real,
        )

    spiral_eig = modes["spiral"][0]
    if spiral_eig.real < 0.0:
        checks["spiral"] = ModeCheck(
            name="spiral", eigenvalues=(spiral_eig,), passed=True, margin=-spiral_eig.real,
        )
    else:
        tau = spiral_eig.time_constant_s
        tau_min = c.stability.spiral_min_time_constant_s
        checks["spiral"] = ModeCheck(
            name="spiral", eigenvalues=(spiral_eig,), passed=tau >= tau_min, margin=tau - tau_min,
        )

    return checks

## TO-DO: add flight quality checks

def evaluate_stability(
    x_np: float, x_cg: float, cref: float, alpha_trim_deg: float,
    wing_incidence_deg: float, eigenvalues: list[Eigenvalue], c: Constants,
) -> StabilityReport:
    """The assignment's stall limit is on the angle of attack *anywhere on the wing*, not the
    body-frame trim alpha: the wing section flies at roughly trim alpha + incidence (there is
    no twist in this model), so that sum is what gets compared against the 10 deg limit."""
    sm = static_margin(x_np, x_cg, cref)
    mode_checks = check_dynamic_modes(eigenvalues, c)
    stall_margin = c.airfoil.stall_aoa_deg - (alpha_trim_deg + wing_incidence_deg)

    return StabilityReport(
        static_margin=sm,
        static_margin_passed=sm > 0.0,
        mode_checks=mode_checks,
        stall_margin_deg=stall_margin,
        stall_margin_passed=stall_margin > 0.0,
    )

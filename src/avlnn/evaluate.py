"""End-to-end: design vector -> AVL run -> objective fitness.

Every caller (EA, dataset generation) should go through `evaluate_design` so the
geometry/mass/run-case/parse round trip stays consistent and AVL failures are handled the same
way everywhere (returned as an infeasible ObjectiveResult rather than raised).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from avlnn.avl_driver import AvlExecutionError, run_avl
from avlnn.config import Constants
from avlnn.geometry import build_avl_geometry
from avlnn.massfile import build_avl_mass_file
from avlnn.objective import ObjectiveResult, evaluate_objective, infeasible_result
from avlnn.parse import AvlParseError, parse_eigenvalues, parse_stability_derivatives
from avlnn.runcase import build_avl_run_case
from avlnn.stability import evaluate_stability


def evaluate_design(design: dict[str, float], c: Constants) -> ObjectiveResult:
    """Runs the full AVL pipeline for one design vector in a scratch temp directory."""
    geometry_text, geom = build_avl_geometry(design, c)
    mass_text = build_avl_mass_file(design, geom, c)
    run_text, _cl_target = build_avl_run_case(design, geom, c)

    with tempfile.TemporaryDirectory(prefix="avlnn_") as tmp:
        try:
            result = run_avl(Path(tmp), geometry_text, mass_text, run_text)
        except AvlExecutionError as exc:
            return infeasible_result(str(exc))

        if result.returncode != 0:
            return infeasible_result(f"AVL exited with code {result.returncode}: {result.stdout[-500:]}")

        if result.eig_text is None:
            return infeasible_result(
                "AVL did not write the eigenvalue (.eig) file -- MODE's eigenmode "
                "calculation likely refused to run; check the tail of AVL stdout"
            )

        try:
            deriv = parse_stability_derivatives(result.stdout)
            eigenvalues = parse_eigenvalues(result.eig_text)
            stability = evaluate_stability(
                x_np=deriv.x_np, x_cg=geom.x_cg_b_m, cref=geom.cref_m,
                alpha_trim_deg=deriv.alpha_deg,
                wing_incidence_deg=design["wing_incidence_deg"],
                eigenvalues=eigenvalues, c=c,
            )
        except (AvlParseError, ValueError) as exc:
            return infeasible_result(str(exc))

        return evaluate_objective(deriv, stability, design, geom.sref_m2, c)

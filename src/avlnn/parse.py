"""Parses AVL output into structured data: the 'ST' stability-derivative dump (from stdout)
and the eigenvalue file written by MODE's W command.

Both formats were verified against a live AVL 3.52 run on 2026-07-15:
- ST stdout: "Alpha =", "CLtot =", "CLa =", "Cma =", "Neutral point  Xnp =" all match.
- Eigenvalues: parsed from the .eig file (three columns: run-case index, real, imag; see
  EIGOUT in AVL's amode.f), not from stdout -- the on-screen MODE output interleaves
  eigenvectors and menu redraws and is not worth scraping.
"""
from __future__ import annotations

import dataclasses
import math
import re


class AvlParseError(RuntimeError):
    """Raised when expected AVL output content could not be found/parsed."""


@dataclasses.dataclass(frozen=True)
class StabilityDerivatives:
    cl: float
    cd: float
    cm: float
    alpha_deg: float          # trimmed angle of attack for this run case
    cl_alpha: float          # CLa: dCL/dalpha (per radian, as AVL reports it)
    cm_alpha: float          # Cma: dCm/dalpha
    x_np: float              # neutral point x-location (same units/frame as Xref)

    @property
    def dcm_dcl(self) -> float:
        return self.cm_alpha / self.cl_alpha


_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_stability_derivatives(text: str) -> StabilityDerivatives:
    def find(pattern: str, label: str, flags: int = 0) -> float:
        m = re.search(pattern, text, flags)
        if m is None:
            raise AvlParseError(f"could not find {label!r} in AVL ST output")
        return float(m.group(1))

    cl = find(rf"\bCLtot\s*=\s*({_FLOAT})", "CLtot")
    cd = find(rf"\bCDtot\s*=\s*({_FLOAT})", "CDtot")
    cm = find(rf"\bCmtot\s*=\s*({_FLOAT})", "Cmtot")
    alpha_deg = find(rf"\bAlpha\s*=\s*({_FLOAT})", "Alpha", re.IGNORECASE)
    cl_alpha = find(rf"\bCLa\s*=\s*({_FLOAT})", "CLa")
    cm_alpha = find(rf"\bCma\s*=\s*({_FLOAT})", "Cma")
    x_np = find(rf"Xnp\s*=\s*({_FLOAT})", "Xnp")

    return StabilityDerivatives(
        cl=cl, cd=cd, cm=cm, alpha_deg=alpha_deg,
        cl_alpha=cl_alpha, cm_alpha=cm_alpha, x_np=x_np,
    )


@dataclasses.dataclass(frozen=True)
class Eigenvalue:
    real: float
    imag: float

    @property
    def is_real_root(self) -> bool:
        return abs(self.imag) < 1e-6

    @property
    def frequency_rad_s(self) -> float:
        return abs(self.imag)

    @property
    def is_stable(self) -> bool:
        return self.real < 0.0

    @property
    def time_constant_s(self) -> float:
        """1/|real part| -- meaningful for real (non-oscillatory) roots only."""
        if self.real == 0.0:
            return math.inf
        return 1.0 / abs(self.real)


# .eig file line (see EIGOUT in AVL's amode.f): run-case index, real part, imag part.
# Fortran list output, e.g. "       1    0.92339096E-02     0.0000000"
_EIG_FILE_LINE = re.compile(
    rf"^\s*(\d+)\s+({_FLOAT})\s+({_FLOAT})\s*$", re.MULTILINE,
)


def parse_eigenvalues(eig_text: str, run_case: int = 1) -> list[Eigenvalue]:
    """Extracts the eigenvalues for `run_case` from the .eig file MODE's W command writes."""
    matches = _EIG_FILE_LINE.findall(eig_text)
    eigenvalues = [
        Eigenvalue(real=float(r), imag=float(i))
        for case, r, i in matches
        if int(case) == run_case
    ]
    if not eigenvalues:
        raise AvlParseError(
            f"no eigenvalues for run case {run_case} in .eig output; "
            "either MODE's eigenmode calculation did not run (check mass/inertia were "
            "MSET-applied) or the file format changed -- see module docstring"
        )
    return eigenvalues

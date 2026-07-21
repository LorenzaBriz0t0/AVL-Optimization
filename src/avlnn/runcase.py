"""Builds an AVL .run file that trims alpha to the cruise CL and elevator to zero pitching
moment about the CG, at the cruise Mach number and altitude.

Format verified against run files written by AVL 3.52 itself (runs/*.run in its source tree)
and against what its parser accepted/rejected in a live session on 2026-07-15:

- Controls not used for trim are constrained to their own deflection ("rudder -> rudder = 0"),
  exactly as AVL-written files do. Moment-constraint spellings are finicky ("Cm pitchmom" is
  accepted, "Cn yaw mom" was not), and in symmetric cruise the roll/yaw moments are zero by
  symmetry anyway.
- Mass and inertia lines are deliberately omitted: the driver applies the .mass file via MSET
  *after* loading this run case, which fills in the true mass/inertia/CG. Writing zeros here
  instead would make amode.f refuse the eigenmode calculation (it requires Ixx,Iyy,Izz > 0).
- Control deflections must NOT appear in the parameter block (AVL: "Ignoring unrecognized
  parameter: aileron"), and the cross-inertia keyword is "Izx", not "Ixz".
"""
from __future__ import annotations

from avlnn.atmosphere import isa
from avlnn.config import Constants
from avlnn.derived import AircraftGeometry, cruise_cl


def build_avl_run_case(
    design: dict[str, float], geom: AircraftGeometry, c: Constants,
) -> tuple[str, float]:
    """Returns (run_file_text, cruise_cl_target)."""
    atm = isa(c.mission.altitude_cruise_m, c.atmosphere)
    velocity = c.mission.mach_cruise * atm.speed_of_sound_m_s
    weight_N = c.mass.total_kg * c.atmosphere.g0_m_s2
    q = 0.5 * atm.density_kg_m3 * velocity**2
    cl_target = cruise_cl(weight_N, q, geom.sref_m2)

    lines = [
        " ---------------------------------------------",
        " Run case  1:   Cruise",
        "",
        f" alpha        ->  CL          =   {cl_target:.5f}",
        " beta         ->  beta        =   0.00000",
        " pb/2V        ->  pb/2V       =   0.00000",
        " qc/2V        ->  qc/2V       =   0.00000",
        " rb/2V        ->  rb/2V       =   0.00000",
        " aileron      ->  aileron     =   0.00000",
        " elevator     ->  Cm pitchmom =   0.00000",
        " rudder       ->  rudder      =   0.00000",
        "",
        " alpha     =   0.00000     deg",
        " beta      =   0.00000     deg",
        " pb/2V     =   0.00000",
        " qc/2V     =   0.00000",
        " rb/2V     =   0.00000",
        f" CL        =   {cl_target:.5f}",
        f" CDo       =   {c.aero.cd0:.5f}",
        " bank      =   0.00000     deg",
        " elevation =   0.00000     deg",
        " heading   =   0.00000     deg",
        f" Mach      =   {c.mission.mach_cruise:.3f}",
        f" velocity  =   {velocity:.5f}     m/s",
        f" density   =   {atm.density_kg_m3:.5f}     kg/m^3",
        f" grav.acc. =   {c.atmosphere.g0_m_s2:.5f}     m/s^2",
        " turn_rad. =   0.00000     m",
        " load_fac. =   1.00000",
        f" X_cg      =   {geom.x_cg_b_m:.5f}     Lunit",
        " Y_cg      =   0.00000     Lunit",
        " Z_cg      =   0.00000     Lunit",
        "",
    ]
    return "\n".join(lines), cl_target

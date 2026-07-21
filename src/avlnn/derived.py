"""Pure geometry/mass math shared by geometry.py and massfile.py.

Modeling assumption (the assignment does not give an explicit x_b for the wing point masses,
only y_b = +-b/6, z_b = 0): each wing's point mass is placed at the local quarter-chord of the
wing planform at that spanwise station, which is the conventional stand-in for a wing's
structural/mass centroid. Because the two wing masses sit at the same x_b (only the sign of
y_b differs) and the 5 fuselage rows are symmetric about x_b=0 by construction, the aircraft
CG is a simple closed-form function of that one point -- no AVL run is needed to find it.
"""
from __future__ import annotations

import dataclasses
import math

from avlnn.config import Constants


@dataclasses.dataclass(frozen=True)
class WingGeometry:
    span_m: float
    root_chord_m: float
    tip_chord_m: float
    mean_aero_chord_m: float
    le_sweep_rad: float
    tip_x_offset_m: float          # tip LE x-offset from root LE, from quarter-chord sweep
    tip_z_offset_m: float          # tip z-offset from root LE, from dihedral
    mass_point_y_m: float          # y_b = +-b/6
    mass_point_x_b_m: float        # x_b of the local quarter-chord at y = b/6 (root-LE-relative)


def wing_geometry(design: dict[str, float]) -> WingGeometry:
    S = design["wing_area_m2"]
    AR = design["wing_aspect_ratio"]
    taper = design["wing_taper_ratio"]
    sweep_c4 = math.radians(design["wing_sweep_c4_deg"])
    dihedral = math.radians(design["wing_dihedral_deg"])

    b = math.sqrt(S * AR)
    c_root = 2.0 * S / (b * (1.0 + taper))
    c_tip = taper * c_root
    mac = (2.0 / 3.0) * c_root * (1.0 + taper + taper**2) / (1.0 + taper)

    # Convert quarter-chord sweep to leading-edge sweep (standard swept-planform relation).
    tan_le = math.tan(sweep_c4) + (1.0 / AR) * (1.0 - taper) / (1.0 + taper)
    le_sweep = math.atan(tan_le)

    y_tip = b / 2.0
    x_tip_offset = y_tip * math.tan(le_sweep)
    z_tip_offset = y_tip * math.tan(dihedral)

    y_mass = b * design.get("wing_point_span_frac", 1.0 / 6.0)
    chord_at_mass = c_root + (c_tip - c_root) * (y_mass / y_tip)
    x_le_at_mass = y_mass * math.tan(le_sweep)
    x_mass_b = x_le_at_mass + 0.25 * chord_at_mass

    return WingGeometry(
        span_m=b, root_chord_m=c_root, tip_chord_m=c_tip, mean_aero_chord_m=mac,
        le_sweep_rad=le_sweep, tip_x_offset_m=x_tip_offset, tip_z_offset_m=z_tip_offset,
        mass_point_y_m=y_mass, mass_point_x_b_m=x_mass_b,
    )


@dataclasses.dataclass(frozen=True)
class TailGeometry:
    area_m2: float
    span_m: float
    root_chord_m: float
    le_sweep_rad: float
    tip_x_offset_m: float
    ac_x_b_m: float          # x_b of this tail's own aerodynamic center (~its quarter chord)
    root_le_x_b_m: float     # x_b of the tail root leading edge


def _tail_from_volume_coeff(
    volume_coeff: float, ref_area: float, ref_length: float, arm_m: float,
    aspect_ratio: float, sweep_c4_deg: float, wing_x_cg_b_m: float,
) -> TailGeometry:
    """Sizes a rectangular (taper=1) tail surface from its volume coefficient and moment arm.

    `ref_length` is cref (horizontal tail) or b (vertical tail), matching the standard
    V_H = S_h*l_h/(S_w*cref) and V_V = S_v*l_v/(S_w*b) tail-volume definitions.
    """
    area = volume_coeff * ref_area * ref_length / arm_m
    span = math.sqrt(area * aspect_ratio)
    root_chord = area / span  # rectangular planform (taper = 1)

    sweep_c4 = math.radians(sweep_c4_deg)
    tan_le = math.tan(sweep_c4)  # taper = 1 => LE sweep == c/4 sweep
    le_sweep = math.atan(tan_le)
    tip_x_offset = (span / 2.0) * math.tan(le_sweep)

    ac_x_b = wing_x_cg_b_m + arm_m
    root_le_x_b = ac_x_b - 0.25 * root_chord

    return TailGeometry(
        area_m2=area, span_m=span, root_chord_m=root_chord, le_sweep_rad=le_sweep,
        tip_x_offset_m=tip_x_offset, ac_x_b_m=ac_x_b, root_le_x_b_m=root_le_x_b,
    )


def htail_geometry(design: dict[str, float], wing: WingGeometry, x_cg_b_m: float) -> TailGeometry:
    return _tail_from_volume_coeff(
        volume_coeff=design["htail_volume_coeff"], ref_area=design["wing_area_m2"],
        ref_length=wing.mean_aero_chord_m, arm_m=design["htail_arm_m"],
        aspect_ratio=design["htail_aspect_ratio"], sweep_c4_deg=design["htail_sweep_c4_deg"],
        wing_x_cg_b_m=x_cg_b_m,
    )


def vtail_geometry(design: dict[str, float], wing: WingGeometry, x_cg_b_m: float) -> TailGeometry:
    return _tail_from_volume_coeff(
        volume_coeff=design["vtail_volume_coeff"], ref_area=design["wing_area_m2"],
        ref_length=wing.span_m, arm_m=design["vtail_arm_m"],
        aspect_ratio=design["vtail_aspect_ratio"], sweep_c4_deg=design["vtail_sweep_c4_deg"],
        wing_x_cg_b_m=x_cg_b_m,
    )


def aircraft_cg_x_b(design: dict[str, float], wing: WingGeometry, c: Constants) -> float:
    """x_b of the CG. Fuselage rows contribute 0 by construction (symmetric about x_b=0);
    only the (equal, same-x_b) left/right wing point masses shift the CG."""
    wing_mass_frac_total = 2.0 * c.mass.wing_fraction_each
    return wing_mass_frac_total * (design["wing_root_le_x_b_m"] + wing.mass_point_x_b_m)


@dataclasses.dataclass(frozen=True)
class AircraftGeometry:
    wing: WingGeometry
    htail: TailGeometry
    vtail: TailGeometry
    x_cg_b_m: float
    sref_m2: float
    cref_m: float
    bref_m: float


def assemble(design: dict[str, float], c: Constants) -> AircraftGeometry:
    """Single entry point that resolves a design vector into full aircraft geometry + CG,
    used by geometry.py, massfile.py, and runcase.py so the three stay consistent."""
    wing = wing_geometry(design)
    x_cg = aircraft_cg_x_b(design, wing, c)
    htail = htail_geometry(design, wing, x_cg)
    vtail = vtail_geometry(design, wing, x_cg)
    return AircraftGeometry(
        wing=wing, htail=htail, vtail=vtail, x_cg_b_m=x_cg,
        sref_m2=design["wing_area_m2"], cref_m=wing.mean_aero_chord_m, bref_m=wing.span_m,
    )


def oswald_efficiency(aspect_ratio: float) -> float:
    return 1.78 * (1.0 - 0.045 * aspect_ratio**0.68) - 0.64


def cruise_cl(weight_N: float, dynamic_pressure_Pa: float, wing_area_m2: float) -> float:
    return weight_N / (dynamic_pressure_Pa * wing_area_m2)


def drag_coefficient(cl: float, aspect_ratio: float, cd0: float) -> float:
    e_o = oswald_efficiency(aspect_ratio)
    return cd0 + cl**2 / (math.pi * e_o * aspect_ratio)


def available_thrust_N(
    static_thrust_to_weight_sl: float, weight_N: float, mach: float, thrust_zero_mach: float,
    rho_at_altitude: float, rho_sea_level: float,
) -> float:
    T0 = static_thrust_to_weight_sl * weight_N
    return T0 * max(0.0, 1.0 - mach / thrust_zero_mach) * (rho_at_altitude / rho_sea_level)

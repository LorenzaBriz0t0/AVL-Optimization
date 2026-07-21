"""Typed loaders for config/aircraft_constants.yaml and config/design_space.yaml."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclasses.dataclass(frozen=True)
class MissionConstants:
    mach_cruise: float
    altitude_cruise_m: float


@dataclasses.dataclass(frozen=True)
class MassConstants:
    total_kg: float
    fuselage_fraction: float
    wing_fraction_each: float
    fuselage_point_mass_kg: float
    fuselage_n_rows: int
    fuselage_points_per_row: int
    fuselage_mid_row_index: int
    d_fuse_m: float
    wing_point_span_frac: float


@dataclasses.dataclass(frozen=True)
class AirfoilConstants:
    wing_section: str
    stall_aoa_deg: float


@dataclasses.dataclass(frozen=True)
class PropulsionConstants:
    static_thrust_to_weight_sl: float
    thrust_zero_mach: float


@dataclasses.dataclass(frozen=True)
class AeroConstants:
    cd0: float


@dataclasses.dataclass(frozen=True)
class TailConstants:
    airfoil: str
    hinge_chord_frac: float
    control_span_frac: float


@dataclasses.dataclass(frozen=True)
class StabilityConstants:
    spiral_min_time_constant_s: float


@dataclasses.dataclass(frozen=True)
class AtmosphereConstants:
    g0_m_s2: float
    R_air_J_per_kgK: float
    gamma_air: float
    T0_sea_level_K: float
    p0_sea_level_Pa: float
    rho0_sea_level_kg_m3: float
    lapse_rate_tropo_K_per_m: float


@dataclasses.dataclass(frozen=True)
class Constants:
    mission: MissionConstants
    mass: MassConstants
    airfoil: AirfoilConstants
    propulsion: PropulsionConstants
    aero: AeroConstants
    tail: TailConstants
    stability: StabilityConstants
    atmosphere: AtmosphereConstants

    @classmethod
    def load(cls, path: Path | None = None) -> "Constants":
        data = _load_yaml(path or CONFIG_DIR / "aircraft_constants.yaml")
        return cls(
            mission=MissionConstants(**data["mission"]),
            mass=MassConstants(**data["mass"]),
            airfoil=AirfoilConstants(**data["airfoil"]),
            propulsion=PropulsionConstants(**data["propulsion"]),
            aero=AeroConstants(**data["aero"]),
            tail=TailConstants(**data["tail"]),
            stability=StabilityConstants(**data["stability"]),
            atmosphere=AtmosphereConstants(**data["atmosphere"]),
        )

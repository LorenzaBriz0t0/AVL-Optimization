"""International Standard Atmosphere (ISA) model.

Only the troposphere (0-11 km) is needed since cruise altitude is exactly 11.0 km (the
tropopause), but the isothermal layer just above it is included too so the optimizer can
explore altitudes infinitesimally past 11 km without the model blowing up.
"""
from __future__ import annotations

import dataclasses
import math

from avlnn.config import AtmosphereConstants

TROPOPAUSE_ALTITUDE_M = 11000.0


@dataclasses.dataclass(frozen=True)
class AtmosphereState:
    altitude_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float


def isa(altitude_m: float, c: AtmosphereConstants) -> AtmosphereState:
    """Temperature, pressure, density, and speed of sound at the given geopotential altitude."""
    if altitude_m < 0:
        raise ValueError(f"altitude must be >= 0 m, got {altitude_m}")

    T0, p0, L, R, g0 = (
        c.T0_sea_level_K, c.p0_sea_level_Pa, c.lapse_rate_tropo_K_per_m,
        c.R_air_J_per_kgK, c.g0_m_s2,
    )
    T_trop = T0 + L * TROPOPAUSE_ALTITUDE_M
    p_trop = p0 * (T_trop / T0) ** (-g0 / (L * R))

    if altitude_m <= TROPOPAUSE_ALTITUDE_M:
        T = T0 + L * altitude_m
        p = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        T = T_trop
        p = p_trop * math.exp(-g0 * (altitude_m - TROPOPAUSE_ALTITUDE_M) / (R * T_trop))

    rho = p / (R * T)
    a = math.sqrt(c.gamma_air * R * T)
    return AtmosphereState(altitude_m, T, p, rho, a)

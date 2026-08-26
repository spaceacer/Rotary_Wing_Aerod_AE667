"""
atmosphere.py
-------------
ISA (International Standard Atmosphere) model up to the tropopause (11 000 m).
Covers BEMT use-cases up to ~7 000 m service ceiling.
"""

import math
from typing import Tuple


def isa_atmosphere(altitude_m: float, dt_k: float = 0.0) -> Tuple[float, float, float]:
    """
    Standard ISA Atmosphere up to the tropopause (11 000 m).

    Parameters
    ----------
    altitude_m : geometric altitude [m]
    dt_k       : temperature offset (ISA deviation) [K]

    Returns
    -------
    (density [kg/m³], speed_of_sound [m/s], temperature [K])
    """
    T0 = 288.15 + dt_k          # sea-level temperature [K]
    P0 = 101_325.0               # sea-level pressure [Pa]
    lapse = -0.0065              # temperature lapse rate [K/m]
    gamma = 1.4                  # ratio of specific heats
    R_air = 287.058              # specific gas constant for air [J/(kg·K)]
    g = 9.80665                  # gravitational acceleration [m/s²]

    if altitude_m <= 11_000.0:
        T = T0 + lapse * altitude_m
        P = P0 * ((T - dt_k) / 288.15) ** (-g / (lapse * R_air))
    else:
        T11k = T0 + lapse * 11_000.0
        P11k = P0 * ((T11k - dt_k) / 288.15) ** (-g / (lapse * R_air))
        T = T11k
        P = P11k * math.exp(-g * (altitude_m - 11_000.0) / (R_air * T))

    rho = P / (R_air * T)
    a = math.sqrt(gamma * R_air * T)
    return rho, a, T

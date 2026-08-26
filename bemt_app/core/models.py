"""
models.py
---------
Dataclasses for RotorGeometry, FlightCondition, and BEMTResult.
These are pure data containers — no UI or computation logic here.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RotorGeometry:
    """Geometric description of a rotor."""
    radius: float                         # Rotor radius R [m]
    root_cutout: float                    # Root cutout radius R_rc [m]
    num_blades: int                       # Number of blades b [-]
    chord_func: Callable[[float], float]  # Local chord distribution c(r) [m]
    twist_func: Callable[[float], float]  # Local pitch distribution theta(r) [rad]


@dataclass
class FlightCondition:
    """Flight / atmospheric operating condition."""
    v_axial: float                        # Axial/climb/cruise velocity V [m/s]
    rpm: float                            # Rotational speed [RPM]
    rho: float = 1.225                    # Air density [kg/m³]
    speed_of_sound: float = 340.29        # Speed of sound a_inf [m/s]

    @property
    def omega(self) -> float:
        """Angular velocity [rad/s]."""
        return self.rpm * (2.0 * np.pi / 60.0)


@dataclass
class BEMTResult:
    """Full output of a BEMT analysis pass."""
    # Integrated performance
    thrust: float
    torque: float
    power: float
    ct: float
    cq: float
    cp: float
    solidity: float
    figure_of_merit: float
    propulsive_eff: float
    stall_fraction: float
    tip_mach: float

    # Spanwise distributions
    r_stations: np.ndarray
    dr: np.ndarray
    inflow_ratio: np.ndarray
    lambda_i: np.ndarray
    lambda_c: float
    phi: np.ndarray             # Inflow angle [rad]
    alpha: np.ndarray           # Angle of attack [rad]
    re_r: np.ndarray            # Reynolds number
    cl: np.ndarray
    cd: np.ndarray
    d_thrust: np.ndarray
    d_torque: np.ndarray
    d_power: np.ndarray
    stalled_mask: np.ndarray

"""
geometry_helpers.py
-------------------
Small geometry utility functions shared between the BEMT solver
and the Streamlit dashboard page.
"""

from typing import Callable, Tuple

import numpy as np


def make_chord_func(
    c_root: float,
    taper: float,
    radius: float,
    root_cutout: float,
) -> Callable[[float], float]:
    """
    Linear chord distribution from root to tip.

    Parameters
    ----------
    c_root    : chord at the root [m]
    taper     : tip/root chord ratio (>1 ⇒ wider at tip, <1 ⇒ tapered)
    radius    : rotor radius [m]
    root_cutout : root cutout radius [m]

    Returns
    -------
    Callable c(r) → chord [m]
    """
    c_tip = c_root * taper
    span = max(radius - root_cutout, 1e-4)
    def _chord(r: float) -> float:
        return c_root + (c_tip - c_root) * ((r - root_cutout) / span)
    return _chord


def make_twist_func(
    theta0_deg: float,
    twist_deg: float,
    radius: float,
) -> Callable[[float], float]:
    """
    Linear pitch distribution: θ(r) = θ₀ + θ_tw · (r/R).

    Parameters
    ----------
    theta0_deg : collective pitch at root [deg]
    twist_deg  : linear twist rate [deg] (negative ⇒ washout)
    radius     : rotor radius [m]

    Returns
    -------
    Callable theta(r) → pitch [rad]
    """
    def _twist(r: float) -> float:
        return np.radians(theta0_deg + twist_deg * (r / max(radius, 1e-4)))
    return _twist


def extract_mean_camber(
    x_af: np.ndarray,
    y_af: np.ndarray,
    num_pts: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the mean camber line from upper/lower surface coordinates.

    Assumes a Selig-format airfoil where coordinates run from trailing edge
    around the top surface to the leading edge, then back to the trailing edge.

    Returns
    -------
    (x_common, y_camber) both of length `num_pts`
    """
    idx_le = int(np.argmin(x_af))
    x_u, y_u = x_af[: idx_le + 1], y_af[: idx_le + 1]
    x_l, y_l = x_af[idx_le:], y_af[idx_le:]

    sort_u = np.argsort(x_u)
    sort_l = np.argsort(x_l)

    x_common = np.linspace(0.0, 1.0, num_pts)
    yu_interp = np.interp(x_common, x_u[sort_u], y_u[sort_u])
    yl_interp = np.interp(x_common, x_l[sort_l], y_l[sort_l])
    yc = 0.5 * (yu_interp + yl_interp)
    return x_common, yc

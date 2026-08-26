"""
bemt_solver.py
--------------
Blade Element Momentum Theory solver.

Implements:
  * Prandtl tip-loss factor
  * Iterative blade-element inflow solver (scalar — for reference)
  * Vectorised full-rotor BEMT solver (run_bemt)
"""

from typing import Tuple

import numpy as np

from .models import BEMTResult, FlightCondition, RotorGeometry


# ---------------------------------------------------------------------------
# Tip-loss
# ---------------------------------------------------------------------------

def prandtl_tip_loss(r_norm: float, lambda_val: float, num_blades: int) -> float:
    """Prandtl tip-loss factor  F(r) = (2/π) · arccos(exp(−f))."""
    if r_norm >= 1.0 or abs(lambda_val) < 1e-6:
        return 1e-4
    f = (num_blades / 2.0) * (1.0 - r_norm) / abs(lambda_val)
    f = np.clip(f, 0.0, 50.0)
    arg = np.clip(np.exp(-f), 0.0, 1.0)
    return max((2.0 / np.pi) * np.arccos(arg), 1e-4)


# ---------------------------------------------------------------------------
# Scalar inflow solver (kept for completeness / unit testing)
# ---------------------------------------------------------------------------

def solve_element_inflow(
    r: float,
    r_norm: float,
    chord: float,
    theta: float,
    geom: RotorGeometry,
    cond: FlightCondition,
    airfoil,
    max_iter: int = 150,
    tol: float = 1e-6,
) -> Tuple[float, float, float, float, float, bool]:
    """
    Iterative scalar inflow solver that matches momentum theory with BET.

    Returns
    -------
    (lambda_total, phi, alpha, cl, cd, is_stalled)
    """
    v_tip = cond.omega * geom.radius
    lambda_c = cond.v_axial / v_tip
    sigma_local = (geom.num_blades * chord) / (np.pi * geom.radius)
    mach_local = (cond.omega * r) / cond.speed_of_sound
    cl_slope = airfoil.cl_slope_fallback

    lambda_i = np.sqrt(
        max(0.0, (sigma_local * cl_slope / 16.0) ** 2
            + (sigma_local * cl_slope / 8.0) * theta * r_norm)
    ) - (sigma_local * cl_slope / 16.0)
    lambda_total = lambda_c + max(lambda_i, 1e-5)

    phi, alpha, cl, cd = 0.0, 0.0, 0.0, 0.0
    is_stalled = False

    for _ in range(max_iter):
        f_loss = prandtl_tip_loss(r_norm, lambda_total, geom.num_blades)
        u_t = cond.omega * r
        u_p = lambda_total * v_tip
        phi = np.arctan2(u_p, u_t)
        alpha = theta - phi
        cl, cd, is_stalled = airfoil.evaluate(alpha, mach_local)

        term1 = (sigma_local * cl_slope) / (16.0 * f_loss) - (lambda_c / 2.0)
        term2 = (sigma_local * cl_slope * theta * r_norm) / (8.0 * f_loss)
        discriminant = max(0.0, term1 ** 2 + term2)
        lambda_new = np.sqrt(discriminant) - term1 + lambda_c

        lambda_next = 0.5 * lambda_total + 0.5 * lambda_new
        if abs(lambda_next - lambda_total) < tol:
            lambda_total = lambda_next
            break
        lambda_total = lambda_next

    return lambda_total, phi, alpha, cl, cd, is_stalled


# ---------------------------------------------------------------------------
# Vectorised full-rotor solver
# ---------------------------------------------------------------------------

def run_bemt(
    geom: RotorGeometry,
    cond: FlightCondition,
    airfoil,
    num_elements: int = 40,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> BEMTResult:
    """
    Run the full Blade Element Momentum Theory analysis for one rotor.

    Uses midpoint stations to avoid the tip singularity at r/R = 1.

    Parameters
    ----------
    geom         : rotor geometry
    cond         : flight condition
    airfoil      : AirfoilModel instance
    num_elements : number of radial strips
    max_iter     : maximum BEMT iteration count
    tol          : convergence tolerance on inflow ratio

    Returns
    -------
    BEMTResult with integrated and spanwise quantities
    """
    r_edges = np.linspace(geom.root_cutout, geom.radius, num_elements + 1)
    r_stations = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    r_norm = r_stations / geom.radius

    chords = np.array([geom.chord_func(r) for r in r_stations])
    thetas = np.array([geom.twist_func(r) for r in r_stations])

    v_tip = cond.omega * geom.radius
    lambda_c = cond.v_axial / v_tip
    sigma_r = (geom.num_blades * chords) / (np.pi * geom.radius)
    mach_r = (cond.omega * r_stations) / cond.speed_of_sound

    # Sutherland viscosity at ISA sea-level (used for Reynolds number)
    T_air = 288.15
    mu_air = 1.458e-6 * T_air ** 1.5 / (T_air + 110.4)
    cl_slope = airfoil.cl_slope_fallback

    # Initial inflow guess
    lambda_i = (
        np.sign(thetas)
        * np.sqrt(np.abs(
            (sigma_r * cl_slope / 16.0) ** 2
            + (sigma_r * cl_slope * thetas * r_norm / 8.0)
        ))
        - (sigma_r * cl_slope / 16.0)
    )
    lambda_total = lambda_c + np.clip(lambda_i, -0.2, 0.5)

    # Iterative BEMT loop
    for _ in range(max_iter):
        f = (geom.num_blades / 2.0) * (1.0 - r_norm) / np.maximum(np.abs(lambda_total), 1e-4)
        f = np.clip(f, 0.0, 40.0)
        f_loss = np.maximum((2.0 / np.pi) * np.arccos(np.exp(-f)), 1e-3)

        u_t = cond.omega * r_stations
        u_p = lambda_total * v_tip
        phi = np.arctan2(u_p, u_t)
        alpha = thetas - phi

        term1 = (sigma_r * cl_slope) / (16.0 * f_loss)
        term2 = (sigma_r * cl_slope * thetas * r_norm) / (8.0 * f_loss)
        disc = term1 ** 2 + np.abs(term2)
        lambda_new = np.sign(thetas) * (np.sqrt(np.maximum(1e-6, disc)) - term1) + lambda_c

        lambda_next = 0.6 * lambda_total + 0.4 * lambda_new
        if np.max(np.abs(lambda_next - lambda_total)) < tol:
            lambda_total = lambda_next
            break
        lambda_total = lambda_next

    # Aerodynamic resultant forces
    u_res_sq = (cond.omega * r_stations) ** 2 + (lambda_total * v_tip) ** 2
    u_res = np.sqrt(u_res_sq)
    re_r = (cond.rho * u_res * chords) / mu_air

    try:
        # BlendedAirfoil takes r_norm to interpolate spanwise
        cl, cd, stalled_mask = airfoil.evaluate_vectorized(alpha, re_r, mach_r, r_norm)
    except TypeError:
        # Fallback for single AirfoilModel (used in Validation page)
        cl, cd, stalled_mask = airfoil.evaluate_vectorized(alpha, re_r, mach_r)

    dt_dr = geom.num_blades * 0.5 * cond.rho * u_res_sq * chords * (
        cl * np.cos(phi) - cd * np.sin(phi)
    )
    dfx_dr = geom.num_blades * 0.5 * cond.rho * u_res_sq * chords * (
        cd * np.cos(phi) + cl * np.sin(phi)
    )
    dq_dr = r_stations * dfx_dr
    dp_dr = cond.omega * dq_dr

    thrust = np.sum(dt_dr * dr)
    torque = np.sum(dq_dr * dr)
    power = np.sum(dp_dr * dr)

    disk_area = np.pi * geom.radius ** 2
    ct = thrust / (cond.rho * disk_area * v_tip ** 2)
    cq = torque / (cond.rho * disk_area * geom.radius * v_tip ** 2)
    cp = power / (cond.rho * disk_area * v_tip ** 3)
    solidity = (geom.num_blades * np.mean(chords)) / (np.pi * geom.radius)

    fm = (ct ** 1.5) / (np.sqrt(2.0) * cp) if (ct > 0 and cp > 0 and cond.v_axial == 0.0) else 0.0
    n_rps = cond.rpm / 60.0
    j_adv = cond.v_axial / (n_rps * 2.0 * geom.radius) if n_rps > 0 else 0.0
    eta_p = (ct * j_adv) / (np.pi * cp) if (cp > 0.0 and cond.v_axial > 0.0) else 0.0

    stall_frac = np.sum(dr[stalled_mask]) / (geom.radius - geom.root_cutout)
    tip_mach = v_tip / cond.speed_of_sound
    lambda_induced = lambda_total - lambda_c

    return BEMTResult(
        thrust=thrust, torque=torque, power=power,
        ct=ct, cq=cq, cp=cp,
        solidity=solidity, figure_of_merit=fm, propulsive_eff=eta_p,
        stall_fraction=stall_frac, tip_mach=tip_mach,
        r_stations=r_stations, dr=dr,
        inflow_ratio=lambda_total, lambda_i=lambda_induced, lambda_c=lambda_c,
        phi=phi, alpha=alpha, re_r=re_r, cl=cl, cd=cd,
        d_thrust=dt_dr, d_torque=dq_dr, d_power=dp_dr,
        stalled_mask=stalled_mask,
    )

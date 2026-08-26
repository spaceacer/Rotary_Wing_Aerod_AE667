"""
airfoil_model.py
----------------
AirfoilModel: loads XFoil polar CSVs, builds a 2-D (alpha, log10(Re))
bivariate spline, and evaluates Cl/Cd with optional Prandtl–Glauert
Mach correction and stall flagging.
"""

import glob
import os
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .airfoil_catalog import (
    ensure_airfoil_data,
    normalize_name,
    parse_single_polar_file,
    to_airfoiltools_slug,
)


class AirfoilModel:
    """
    Airfoil aerodynamic model with 2-D polar interpolation.

    Parameters
    ----------
    airfoil_name:
        Human-readable airfoil name (e.g. "NACA 0012").
        Use "Knight & Hefner Analytical" for the analytical fallback.
    search_dirs:
        Directories to search for cached polar CSV / DAT files.
    cl_slope_fallback:
        Thin-airfoil Cl_alpha [1/rad] used when no polars are available.
    cd0_fallback:
        Zero-lift drag coefficient for the analytical fallback.
    alpha_stall_deg_fallback:
        Stall angle [deg] used for the stall mask.
    ncrit_pref:
        Preferred Ncrit value (5 = turbulent, 9 = standard).
    """

    def __init__(
        self,
        airfoil_name: str = "NACA 0012",
        search_dirs: Tuple[str, ...] = ("airfoils", "."),
        cl_slope_fallback: float = 5.75,
        cd0_fallback: float = 0.0113,
        alpha_stall_deg_fallback: float = 14.0,
        ncrit_pref: Optional[int] = 9,
    ) -> None:
        self.airfoil_name = airfoil_name
        self.search_dirs = search_dirs
        self.cl_slope_fallback = cl_slope_fallback
        self.cd0_fallback = cd0_fallback
        self.alpha_stall = np.radians(alpha_stall_deg_fallback)
        self.has_polars = False
        self.re_list: list = []
        self.raw_data: dict = {}

        is_analytical = airfoil_name in (
            "Knight & Hefner (1937)",
            "Knight & Hefner Analytical",
        )
        if not is_analytical:
            ensure_airfoil_data(self.airfoil_name)
            self._build_polar_interpolators(ncrit_pref)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_polar_interpolators(self, ncrit_pref: Optional[int]) -> None:
        slug = to_airfoiltools_slug(self.airfoil_name)
        slug_base = slug.replace("-il", "").replace("-sa", "")

        all_files: list = []
        for d in self.search_dirs:
            if os.path.exists(d):
                all_files.extend(glob.glob(os.path.join(d, "*.csv")))
                all_files.extend(glob.glob(os.path.join(d, "*.txt")))
                all_files.extend(glob.glob(os.path.join(d, "*", "*.csv")))
                all_files.extend(glob.glob(os.path.join(d, "*", "*.txt")))

        re_data: dict = {}
        for fpath in set(all_files):
            fname = os.path.basename(fpath).lower()
            if fname.startswith("xf-") or "polar" in fname:
                if slug_base in fname or normalize_name(self.airfoil_name) in fname:
                    re_v, nc_v, df = parse_single_polar_file(fpath)
                    if df is not None and re_v is not None:
                        if ncrit_pref == 5 and (nc_v == 5 or "-n5" in fname):
                            re_data[re_v] = df
                        elif ncrit_pref == 9 and (
                            nc_v == 9 or (nc_v != 5 and "-n5" not in fname)
                        ):
                            re_data[re_v] = df
                        elif ncrit_pref is None:
                            re_data[re_v] = df

        if not re_data:
            self.has_polars = False
            return

        sorted_res = sorted(re_data.keys())
        self.re_list = sorted_res
        self.raw_data = re_data

        alpha_common = np.linspace(np.radians(-15.0), np.radians(20.0), 71)
        cl_grid = np.zeros((len(alpha_common), len(sorted_res)))
        cd_grid = np.zeros((len(alpha_common), len(sorted_res)))

        for j, re_v in enumerate(sorted_res):
            df = re_data[re_v]
            alpha_rad = np.radians(df["alpha"].values.astype(float))
            cl_raw = df["CL"].values.astype(float)
            cd_raw = df["CD"].values.astype(float)

            sort_idx = np.argsort(alpha_rad)
            alpha_rad = alpha_rad[sort_idx]
            cl_raw = cl_raw[sort_idx]
            cd_raw = cd_raw[sort_idx]

            cl_interp = np.interp(alpha_common, alpha_rad, cl_raw,
                                  left=cl_raw[0], right=cl_raw[-1])
            cd_interp = np.interp(alpha_common, alpha_rad, cd_raw,
                                  left=cd_raw[0], right=cd_raw[-1])

            # Post-stall drag augmentation
            stall_high = alpha_common > np.radians(14.0)
            stall_low = alpha_common < np.radians(-10.0)
            cd_interp[stall_high] += (
                1.2 * np.sin(alpha_common[stall_high] - np.radians(14.0)) ** 2
            )
            cd_interp[stall_low] += (
                1.2 * np.sin(alpha_common[stall_low] + np.radians(10.0)) ** 2
            )

            cl_grid[:, j] = cl_interp
            cd_grid[:, j] = cd_interp

        log_res = np.log10(sorted_res)
        kx = 2
        ky = min(len(sorted_res) - 1, 2)
        if len(sorted_res) > 1:
            self.spline_cl = RectBivariateSpline(alpha_common, log_res, cl_grid, kx=kx, ky=ky)
            self.spline_cd = RectBivariateSpline(alpha_common, log_res, cd_grid, kx=kx, ky=ky)
        else:
            # Scalar-Re fallback — capture grid reference
            _cl_col = cl_grid[:, 0].copy()
            _cd_col = cd_grid[:, 0].copy()
            _ac = alpha_common.copy()
            self.spline_cl = lambda a, r, _cl=_cl_col, _ac=_ac: np.interp(a, _ac, _cl)  # noqa
            self.spline_cd = lambda a, r, _cd=_cd_col, _ac=_ac: np.interp(a, _ac, _cd)  # noqa

        self.has_polars = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, alpha: float, mach: float = 0.0, re: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """Scalar evaluation for a single blade element."""
        re_target = re if re is not None else (
            float(np.median(self.re_list)) if self.re_list else 2.0e5
        )
        cl_arr, cd_arr, stalled_arr = self.evaluate_vectorized(
            np.array([alpha]), np.array([re_target]), np.array([mach])
        )
        return float(cl_arr[0]), float(cd_arr[0]), bool(stalled_arr[0])

    def evaluate_vectorized(
        self,
        alpha_arr: np.ndarray,
        re_arr: np.ndarray,
        mach_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized 2-D spline evaluation over arrays of alpha and Re."""
        if self.has_polars:
            log_re = np.log10(np.clip(re_arr, 1e4, 1e7))
            cl = self.spline_cl.ev(alpha_arr, log_re)
            cd = self.spline_cd.ev(alpha_arr, log_re)
        else:
            re_ref = 2.4e5
            re_factor = np.clip((re_ref / np.maximum(re_arr, 1e4)) ** 0.2, 0.7, 1.8)
            cl = self.cl_slope_fallback * alpha_arr
            cd = (self.cd0_fallback * re_factor) + 1.25 * alpha_arr**2

        if np.any(mach_arr > 0.0):
            beta = np.sqrt(np.maximum(1e-4, 1.0 - np.clip(mach_arr, 0.0, 0.95) ** 2))
            cl = cl / beta
            cd = cd / beta

        is_stalled = np.abs(alpha_arr) > self.alpha_stall
        return cl, cd, is_stalled

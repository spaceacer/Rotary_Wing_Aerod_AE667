"""
airfoil_blend.py
----------------
Provides the `BlendedAirfoil` class which wraps multiple `AirfoilModel`
instances defined at specific spanwise coordinates (r/R). It interpolates
aerodynamic properties (Cl, Cd) and 3D geometric coordinates.
"""

from typing import List, Tuple

import numpy as np

from .airfoil_catalog import load_airfoil_coords
from .airfoil_model import AirfoilModel
from .geometry_helpers import extract_mean_camber


class BlendedAirfoil:
    def __init__(self, stations: List[float], models: List[AirfoilModel]):
        """
        Parameters
        ----------
        stations : list of r/R coordinates (e.g. [0.2, 1.0])
        models   : list of corresponding AirfoilModel instances
        """
        sorted_pairs = sorted(zip(stations, models), key=lambda x: x[0])
        self.r_norms = np.array([p[0] for p in sorted_pairs])
        self.models = [p[1] for p in sorted_pairs]
        self.has_polars = any(m.has_polars for m in self.models)
        self.cl_slope_fallback = np.mean([m.cl_slope_fallback for m in self.models])

        # Pre-compute unified normalized coordinates for 3D lofting
        self.norm_coords = []
        for model in self.models:
            x_raw, y_raw = load_airfoil_coords(model.airfoil_name)
            self.norm_coords.append(self._normalize_coords(x_raw, y_raw))

    def _normalize_coords(self, x_af: np.ndarray, y_af: np.ndarray, num_pts: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps any airfoil coordinate set onto a standard cosine-spaced x-grid 
        so that multiple airfoils can be linearly blended point-by-point.
        """
        idx_le = int(np.argmin(x_af))
        x_u, y_u = x_af[:idx_le + 1], y_af[:idx_le + 1]
        x_l, y_l = x_af[idx_le:], y_af[idx_le:]

        su = np.argsort(x_u)
        sl = np.argsort(x_l)
        x_u, y_u = x_u[su], y_u[su]
        x_l, y_l = x_l[sl], y_l[sl]

        # Cosine spacing clusters points at LE (0) and TE (1)
        beta = np.linspace(0, np.pi, num_pts)
        x_std = 0.5 * (1.0 - np.cos(beta))

        yu_std = np.interp(x_std, x_u, y_u)
        yl_std = np.interp(x_std, x_l, y_l)

        # Reconstruct Selig format (TE -> LE -> TE)
        x_norm = np.concatenate([x_std[::-1], x_std[1:]])
        y_norm = np.concatenate([yu_std[::-1], yl_std[1:]])
        return x_norm, y_norm

    def get_blended_coords(self, r_norm: float, use_camber: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the interpolated (x, y) coordinates for the local airfoil shape."""
        if r_norm <= self.r_norms[0]:
            x, y = self.norm_coords[0]
        elif r_norm >= self.r_norms[-1]:
            x, y = self.norm_coords[-1]
        else:
            idx = np.searchsorted(self.r_norms, r_norm)
            r0, r1 = self.r_norms[idx - 1], self.r_norms[idx]
            w = (r_norm - r0) / (max(r1 - r0, 1e-6))
            
            x0, y0 = self.norm_coords[idx - 1]
            x1, y1 = self.norm_coords[idx]
            
            x = x0  # x-grids are identical by design
            y = y0 * (1 - w) + y1 * w

        if use_camber:
            return extract_mean_camber(x, y, num_pts=32)
        return x, y

    def evaluate_vectorized(
        self, alpha_arr: np.ndarray, re_arr: np.ndarray, mach_arr: np.ndarray, r_norm_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates Cl, Cd for all defined airfoils and blends the results
        along the span based on r_norm_arr.
        """
        # 1. Evaluate every model across the whole array (fast, vectorized)
        cl_all, cd_all, st_all = [], [], []
        for model in self.models:
            cl, cd, st = model.evaluate_vectorized(alpha_arr, re_arr, mach_arr)
            cl_all.append(cl)
            cd_all.append(cd)
            st_all.append(st)

        cl_all = np.array(cl_all)
        cd_all = np.array(cd_all)
        st_all = np.array(st_all)

        # 2. Blend results based on local r/R (Fully Vectorized)
        r_clipped = np.clip(r_norm_arr, self.r_norms[0], self.r_norms[-1])
        idx = np.searchsorted(self.r_norms, r_clipped)
        idx = np.clip(idx, 1, len(self.r_norms) - 1)

        r0 = self.r_norms[idx - 1]
        r1 = self.r_norms[idx]
        w = (r_clipped - r0) / np.maximum(r1 - r0, 1e-6)

        arr_idx = np.arange(len(r_norm_arr))
        
        cl0, cl1 = cl_all[idx - 1, arr_idx], cl_all[idx, arr_idx]
        cd0, cd1 = cd_all[idx - 1, arr_idx], cd_all[idx, arr_idx]
        st0, st1 = st_all[idx - 1, arr_idx], st_all[idx, arr_idx]

        cl_out = cl0 * (1 - w) + cl1 * w
        cd_out = cd0 * (1 - w) + cd1 * w
        stall_out = np.where(w < 0.5, st0, st1)

        return cl_out, cd_out, stall_out

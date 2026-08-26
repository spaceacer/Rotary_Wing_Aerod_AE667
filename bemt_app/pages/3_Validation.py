"""
pages/3_Validation.py
----------------------
BEMT solver validation against Knight & Hefner (1937) / NACA TN 626
experimental data for a rectangular-blade hovering rotor.

Controls
--------
  Blade count selector (b = 2, 3, 4, 5)

Outputs
-------
  Error metrics table (RMSE, MAE for CT and CP)
  Three-panel plot: CT vs θ₀ | CP vs θ₀ | FM vs CT
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.airfoil_model import AirfoilModel
from core.bemt_solver import run_bemt
from core.geometry_helpers import make_chord_func, make_twist_func
from core.models import FlightCondition, RotorGeometry

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K&H Validation — BEMT",
    page_icon="📊",
    layout="wide",
)

st.title("📊  BEMT Validation vs Knight & Hefner (1937)")
st.markdown(
    """
    Compares the BEMT solver output against digitised experimental data from
    **NACA Technical Note 626** (Knight & Hefner, 1937) for a rectangular-chord
    hovering rotor.

    **Rotor parameters (fixed):**
    - Radius R = 0.762 m
    - Root cut-out R_rc = 0.127 m (5 inches)
    - Chord c = 0.0508 m (2 inches)
    - ρ = 1.225 kg/m³ (sea-level)
    - RPM = 960 rpm
    - Airfoil: NACA 0015 Analytical Model (a₀=5.75, ε=1.25)
    """
)
st.markdown("---")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Configuration")
    B_kh = st.selectbox("Blade count (b)", [2, 3, 4, 5], index=0)
    st.markdown(
        "_Note: Experimental data is available for all configurations (b = 2, 3, 4, 5)._"
    )

# ── constants ─────────────────────────────────────────────────────────────────
R_kh = 0.762
R_rc_kh = 0.127
c_kh = 0.0508
rho_sl = 1.225

# =======================================================================
# 1. NACA TN 626 EXPERIMENTAL DATA (from Tables I - IV)
# =======================================================================
# Note: The raw NACA tables use C_T = T / (0.5 * rho * V_tip^2 * A).
# We multiply by 0.5 to convert to modern C_T = T / (rho * V_tip^2 * A).
exp_data_dict = {
    2: {  
        'theta_deg': np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        'CT': 0.5 * np.array([0.0, 0.000280, 0.000873, 0.00248, 0.00442, 0.00650, 0.00847, 0.00990]),
        'CP': 0.5 * np.array([0.0001080, 0.000111, 0.000125, 0.000191, 0.000316, 0.000494, 0.000691, 0.000878]),
    },
    3: {  
        'theta_deg': np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        'CT': 0.5 * np.array([0.0, 0.00102, 0.00298, 0.00548, 0.00833, 0.01125, 0.01370]),
        'CP': 0.5 * np.array([0.0001850, 0.000206, 0.000300, 0.000474, 0.000735, 0.001048, 0.001357]),
    },
    4: {  
        'theta_deg': np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        'CT': 0.5 * np.array([0.0, 0.000287, 0.001042, 0.00214, 0.00338, 0.00473, 0.00645, 0.00792, 0.00981, 0.01182, 0.01382, 0.01596, 0.01745]),
        'CP': 0.5 * np.array([0.000268, 0.000274, 0.000300, 0.000338, 0.000410, 0.000499, 0.000620, 0.000743, 0.000920, 0.001162, 0.001395, 0.00171, 0.00191]),
    },
    5: {  
        'theta_deg': np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        'CT': 0.5 * np.array([0.0, 0.001181, 0.00362, 0.00694, 0.01103, 0.01548, 0.02000]),
        'CP': 0.5 * np.array([0.0002380, 0.000270, 0.000396, 0.000680, 0.001086, 0.001597, 0.002240]),
    }
}

exp_theta_deg = exp_data_dict[B_kh]['theta_deg']
exp_ct_ref = exp_data_dict[B_kh]['CT']
exp_cp_ref = exp_data_dict[B_kh]['CP']

exp_fm_ref = np.zeros_like(exp_ct_ref)
with np.errstate(divide='ignore', invalid='ignore'):
    valid = (exp_ct_ref > 0) & (exp_cp_ref > 0)
    exp_fm_ref[valid] = (exp_ct_ref[valid] ** 1.5) / (np.sqrt(2.0) * exp_cp_ref[valid])

# ── airfoil (NACA 0015) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NACA 0015 airfoil data …")
def _load_airfoil():
    return AirfoilModel(airfoil_name="Knight & Hefner Analytical")

airfoil_kh = _load_airfoil()
cond_kh = FlightCondition(v_axial=0.0, rpm=960.0, rho=rho_sl)

# ── sweep ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running BEMT sweep …")
def _sweep(num_blades: int, thetas: np.ndarray):
    ct_list, cp_list, fm_list = [], [], []
    for th in thetas:
        geom = RotorGeometry(
            radius=R_kh,
            root_cutout=R_rc_kh,
            num_blades=num_blades,
            chord_func=make_chord_func(c_kh, 1.0, R_kh, R_rc_kh),
            twist_func=make_twist_func(float(th), 0.0, R_kh),
        )
        r = run_bemt(geom, cond_kh, airfoil_kh, num_elements=40)
        ct_list.append(r.ct)
        cp_list.append(r.cp)
        fm_list.append(r.figure_of_merit)
    return np.array(ct_list), np.array(cp_list), np.array(fm_list)


with st.spinner(f"Running BEMT sweep for b = {B_kh} …"):
    bemt_ct, bemt_cp, bemt_fm = _sweep(B_kh, exp_theta_deg)
    plot_theta_deg = np.linspace(0.0, 12.5, 35)
    plot_bemt_ct, plot_bemt_cp, plot_bemt_fm = _sweep(B_kh, plot_theta_deg)

# ── error metrics ─────────────────────────────────────────────────────────────
rmse_ct = float(np.sqrt(np.mean((bemt_ct - exp_ct_ref) ** 2)))
mae_ct = float(np.mean(np.abs(bemt_ct - exp_ct_ref)))
rmse_cp = float(np.sqrt(np.mean((bemt_cp - exp_cp_ref) ** 2)))
mae_cp = float(np.mean(np.abs(bemt_cp - exp_cp_ref)))

st.subheader(f"📐  Error Metrics  (vs b = {B_kh} experimental reference)")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("CT  RMSE", f"{rmse_ct:.6f}")
m_col2.metric("CT  MAE", f"{mae_ct:.6f}")
m_col3.metric("CP  RMSE", f"{rmse_cp:.6f}")
m_col4.metric("CP  MAE", f"{mae_cp:.6f}")

# Tabular comparison
df_cmp = pd.DataFrame(
    {
        "θ₀ [°]": exp_theta_deg,
        f"CT  exp (b={B_kh})": exp_ct_ref,
        f"CT  BEMT (b={B_kh})": bemt_ct,
        "CT  error": bemt_ct - exp_ct_ref,
        f"CP  exp (b={B_kh})": exp_cp_ref,
        f"CP  BEMT (b={B_kh})": bemt_cp,
        "CP  error": bemt_cp - exp_cp_ref,
    }
)
with st.expander("📋  Full comparison table"):
    st.dataframe(
        df_cmp.style.format(
            {
                f"CT  exp (b={B_kh})": "{:.5f}",
                f"CT  BEMT (b={B_kh})": "{:.5f}",
                "CT  error": "{:+.6f}",
                f"CP  exp (b={B_kh})": "{:.6f}",
                f"CP  BEMT (b={B_kh})": "{:.6f}",
                "CP  error": "{:+.6f}",
            }
        ),
        use_container_width=True,
    )

st.markdown("---")

# ── residual calculations ─────────────────────────────────────────────────────
res_ct = bemt_ct - exp_ct_ref
res_cp = bemt_cp - exp_cp_ref
res_fm = np.zeros_like(res_ct)
valid_fm_pts = (exp_ct_ref > 0) & (exp_fm_ref > 0)
res_fm[valid_fm_pts] = bemt_fm[valid_fm_pts] - exp_fm_ref[valid_fm_pts]

# ── plots ─────────────────────────────────────────────────────────────────────
st.subheader("📈  Validation & Residual Analysis Plots")

fig, axs = plt.subplots(2, 3, figsize=(14, 7.8), gridspec_kw={'height_ratios': [1.15, 0.85]})

for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.35, color="gray")

# [0, 0] CT vs θ₀
axs[0, 0].plot(exp_theta_deg, exp_ct_ref, "ko", markersize=6.5,
               label=f"K&H Exp (b={B_kh})", zorder=5)
axs[0, 0].plot(plot_theta_deg, plot_bemt_ct, "b-", lw=2.2,
               label=f"BEMT  (b={B_kh})")
axs[0, 0].set_xlabel("Collective Pitch θ₀ [deg]")
axs[0, 0].set_ylabel("Thrust Coefficient C_T")
axs[0, 0].set_title("C_T  vs  θ₀")
axs[0, 0].legend(fontsize=8)

# [0, 1] CP vs θ₀
axs[0, 1].plot(exp_theta_deg, exp_cp_ref, "ks", markersize=6.5,
               label=f"K&H Exp (b={B_kh})", zorder=5)
axs[0, 1].plot(plot_theta_deg, plot_bemt_cp, "r-", lw=2.2,
               label=f"BEMT  (b={B_kh})")
axs[0, 1].set_xlabel("Collective Pitch θ₀ [deg]")
axs[0, 1].set_ylabel("Power Coefficient C_P")
axs[0, 1].set_title("C_P  vs  θ₀")
axs[0, 1].legend(fontsize=8)

# [0, 2] FM vs CT
valid_plot = plot_bemt_ct > 0
axs[0, 2].plot(exp_ct_ref[valid_fm_pts], exp_fm_ref[valid_fm_pts], "k^", markersize=6.5,
               label=f"K&H Exp (b={B_kh})", zorder=5)
axs[0, 2].plot(plot_bemt_ct[valid_plot], plot_bemt_fm[valid_plot], "g-", lw=2.2,
               label=f"BEMT  (b={B_kh})")
axs[0, 2].set_xlabel("Thrust Coefficient C_T")
axs[0, 2].set_ylabel("Figure of Merit (FM)")
axs[0, 2].set_title("FM  vs  C_T")
axs[0, 2].set_ylim(0.0, 0.85)
axs[0, 2].legend(fontsize=8)

# [1, 0] Delta CT Residuals
axs[1, 0].axhline(0, color="black", linestyle="--", lw=1.0, alpha=0.7)
axs[1, 0].plot(exp_theta_deg, res_ct, "b-o", markersize=5, lw=1.5,
               label=r"$\Delta C_T = C_{T,\mathrm{BEMT}} - C_{T,\mathrm{exp}}$")
axs[1, 0].fill_between(exp_theta_deg, res_ct, 0, color="#4fc3f7", alpha=0.25)
axs[1, 0].set_xlabel("Collective Pitch θ₀ [deg]")
axs[1, 0].set_ylabel(r"$\Delta C_T$ Residual")
axs[1, 0].set_title(r"$C_T$ Residuals ($\Delta C_T$ vs $\theta_0$)")
axs[1, 0].legend(fontsize=7.5)

# [1, 1] Delta CP Residuals
axs[1, 1].axhline(0, color="black", linestyle="--", lw=1.0, alpha=0.7)
axs[1, 1].plot(exp_theta_deg, res_cp, "r-s", markersize=5, lw=1.5,
               label=r"$\Delta C_P = C_{P,\mathrm{BEMT}} - C_{P,\mathrm{exp}}$")
axs[1, 1].fill_between(exp_theta_deg, res_cp, 0, color="tomato", alpha=0.25)
axs[1, 1].set_xlabel("Collective Pitch θ₀ [deg]")
axs[1, 1].set_ylabel(r"$\Delta C_P$ Residual")
axs[1, 1].set_title(r"$C_P$ Residuals ($\Delta C_P$ vs $\theta_0$)")
axs[1, 1].legend(fontsize=7.5)

# [1, 2] Delta FM Residuals
axs[1, 2].axhline(0, color="black", linestyle="--", lw=1.0, alpha=0.7)
axs[1, 2].plot(exp_ct_ref[valid_fm_pts], res_fm[valid_fm_pts], "g-^", markersize=5, lw=1.5,
               label=r"$\Delta \mathrm{FM} = \mathrm{FM}_{\mathrm{BEMT}} - \mathrm{FM}_{\mathrm{exp}}$")
axs[1, 2].fill_between(exp_ct_ref[valid_fm_pts], res_fm[valid_fm_pts], 0, color="lightgreen", alpha=0.3)
axs[1, 2].set_xlabel("Thrust Coefficient C_T")
axs[1, 2].set_ylabel(r"$\Delta \mathrm{FM}$ Residual")
axs[1, 2].set_title(r"$\mathrm{FM}$ Residuals ($\Delta \mathrm{FM}$ vs $C_T$)")
axs[1, 2].legend(fontsize=7.5)

fig.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    "Experimental data sourced from: Knight, M. & Hefner, R. A. (1937). "
    "_Analysis of Ground Effect on the Lifting Airscrew._ NACA TN 626."
)

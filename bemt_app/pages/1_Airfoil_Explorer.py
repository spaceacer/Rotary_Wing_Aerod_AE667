"""
pages/1_Airfoil_Explorer.py
---------------------------
Interactive Cl / Cd / Drag-polar explorer.

Sidebar controls:
  • Airfoil search text box
  • Airfoil dropdown (filtered)
  • Ncrit selector
  • Reynolds number slider

Main area:
  • Three matplotlib charts (Cl vs α, Cd vs α, Drag polar)
  • Raw polar data lines (one per Re dataset, colour-coded)
  • Live interpolated spline curve at the selected Re
  • Status badge (polars loaded / analytical fallback)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── path fix so the core package resolves when the page is run directly ──────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.airfoil_catalog import filter_airfoil_catalog, get_all_available_airfoils
from core.airfoil_model import AirfoilModel

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airfoil Explorer — BEMT",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️  Airfoil Polar 2-D Interpolator Explorer")
st.caption(
    "Select an airfoil and explore its Cl / Cd polars across a range of "
    "Reynolds numbers.  Data is fetched automatically from AirfoilTools.com "
    "and cached locally."
)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading airfoil catalog …")
def load_catalog() -> list:
    return get_all_available_airfoils()


@st.cache_data(show_spinner="Loading airfoil polars …")
def load_model(name: str, ncrit: int) -> AirfoilModel:
    return AirfoilModel(airfoil_name=name, ncrit_pref=ncrit)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Controls")

    all_airfoils = load_catalog()

    search_query = st.text_input(
        "Search airfoils",
        placeholder="e.g. S1223, Clark, NACA 23…",
    )
    filtered = filter_airfoil_catalog(search_query, all_airfoils)

    selected_af = st.selectbox(
        "Airfoil",
        options=filtered if filtered else all_airfoils,
        index=0,
    )

    ncrit = st.selectbox(
        "Ncrit",
        options=[9, 5],
        format_func=lambda v: "9 — Standard (free transition)"
        if v == 9
        else "5 — Turbulent / rough surface",
    )

    target_re = st.slider(
        "Interpolation Re",
        min_value=40_000,
        max_value=1_200_000,
        value=200_000,
        step=10_000,
        format="%d",
        help="Reynolds number at which the live spline curve is evaluated.",
    )

# ── load model ────────────────────────────────────────────────────────────────
model = load_model(selected_af, ncrit)

# Status badge
if model.has_polars:
    re_str = ", ".join(f"{int(r):,}" for r in model.re_list)
    st.success(
        f"**{len(model.re_list)} Re dataset(s) loaded** — "
        f"Re = {re_str}",
        icon="✅",
    )
else:
    st.warning(
        "No polar data found.  Showing **analytical thin-airfoil fallback** "
        "(Cl_α = 5.75 rad⁻¹, Cd₀ = 0.0113).",
        icon="⚠️",
    )

# ── compute spline curve ───────────────────────────────────────────────────────
alpha_sweep = np.linspace(-12.0, 16.0, 200)
alpha_rad = np.radians(alpha_sweep)
re_arr = np.full_like(alpha_rad, float(target_re))
mach_arr = np.zeros_like(alpha_rad)
cl_vals, cd_vals, _ = model.evaluate_vectorized(alpha_rad, re_arr, mach_arr)

# ── plots ─────────────────────────────────────────────────────────────────────
colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(model.re_list), 1)))

fig, (ax_cl, ax_cd, ax_polar) = plt.subplots(1, 3, figsize=(14, 4.5))
for ax in (ax_cl, ax_cd, ax_polar):
    ax.tick_params(labelsize=8)

# Raw polar datasets
if model.has_polars:
    for i, r_val in enumerate(model.re_list):
        df = model.raw_data[r_val]
        kw = dict(linestyle="--", marker="o", markersize=2.5,
                  color=colors[i], label=f"Re = {int(r_val):,}")
        ax_cl.plot(df["alpha"], df["CL"], **kw)
        ax_cd.plot(df["alpha"], df["CD"], **kw)
        ax_polar.plot(df["CD"], df["CL"], **kw)

# Live spline curve
spline_kw = dict(color="tomato", linewidth=2.2, label=f"Spline @ Re={target_re:,}")
ax_cl.plot(alpha_sweep, cl_vals, **spline_kw)
ax_cd.plot(alpha_sweep, cd_vals, **spline_kw)
ax_polar.plot(cd_vals, cl_vals, **spline_kw)

# Formatting
ax_cl.set_xlabel("Angle of Attack α [deg]")
ax_cl.set_ylabel("Lift Coefficient Cl")
ax_cl.set_title(f"Cl vs α  |  {selected_af}")
ax_cl.set_xlim(-12, 16)
ax_cl.legend(fontsize=7)
ax_cl.grid(True, linestyle=":", alpha=0.35, color="gray")

ax_cd.set_xlabel("Angle of Attack α [deg]")
ax_cd.set_ylabel("Drag Coefficient Cd")
ax_cd.set_title("Cd vs α")
ax_cd.set_xlim(-12, 16)
ax_cd.legend(fontsize=7)
ax_cd.grid(True, linestyle=":", alpha=0.35, color="gray")

ax_polar.set_xlabel("Drag Coefficient Cd")
ax_polar.set_ylabel("Lift Coefficient Cl")
ax_polar.set_title("Drag Polar  (Cl vs Cd)")
ax_polar.legend(fontsize=7)
ax_polar.grid(True, linestyle=":", alpha=0.35, color="gray")

fig.tight_layout(pad=1.4)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── raw data table (optional) ─────────────────────────────────────────────────
if model.has_polars:
    with st.expander("📋  Raw polar data tables"):
        for r_val in model.re_list:
            st.markdown(f"**Re = {int(r_val):,}**")
            st.dataframe(
                model.raw_data[r_val].reset_index(drop=True),
                use_container_width=True,
                height=200,
            )

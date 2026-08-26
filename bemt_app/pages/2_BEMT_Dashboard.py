"""
pages/2_BEMT_Dashboard.py
--------------------------
Interactive BEMT rotor analysis dashboard.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.airfoil_catalog import filter_airfoil_catalog, get_all_available_airfoils
from core.airfoil_model import AirfoilModel
from core.airfoil_blend import BlendedAirfoil
from core.bemt_solver import run_bemt
from core.geometry_helpers import (
    make_chord_func,
    make_twist_func,
)
from core.models import FlightCondition, RotorGeometry

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BEMT Dashboard",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁  BEMT Rotor Analysis Dashboard")

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading catalog …")
def _catalog() -> list:
    return get_all_available_airfoils()


@st.cache_data(show_spinner="Loading airfoil model …")
def _model(name: str, ncrit: int) -> AirfoilModel:
    return AirfoilModel(airfoil_name=name, ncrit_pref=ncrit)


# ── 3-D view angle state ──────────────────────────────────────────────────────
if "elev" not in st.session_state:
    st.session_state["elev"] = 20
if "azim" not in st.session_state:
    st.session_state["azim"] = -65

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Parameters")

    st.subheader("Airfoil Stations")
    st.markdown("Define the airfoils along the blade span.")
    all_af = _catalog()
    
    if "airfoil_table" not in st.session_state:
        st.session_state["airfoil_table"] = pd.DataFrame({
            "r/R": [0.2, 1.0],
            "Airfoil": ["Clark Y", "NACA 0012"]
        })
        
    edited_df = st.data_editor(
        st.session_state["airfoil_table"],
        num_rows="dynamic",
        column_config={
            "r/R": st.column_config.NumberColumn(
                "r/R Station", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
            "Airfoil": st.column_config.SelectboxColumn(
                "Airfoil", options=all_af, required=True
            )
        },
        use_container_width=True,
        hide_index=True
    )
    st.session_state["airfoil_table"] = edited_df
    
    ncrit = st.selectbox(
        "Ncrit",
        [9, 5],
        format_func=lambda v: "9 — Standard" if v == 9 else "5 — Turbulent",
    )

    st.subheader("Blade Geometry & Twist")
    radius = st.slider("Radius R [m]", 0.20, 2.50, 0.762, 0.01)
    pitch = st.slider("Collective pitch θ₀ [°]", 0.0, 25.0, 8.0, 0.5)
    twist = st.slider("Linear twist θ_tw [°]", -35.0, 10.0, 0.0, 0.5)
    c_root = st.slider("Root chord [m]", 0.010, 0.150, 0.0508, 0.005)
    taper = st.slider("Taper (c_tip / c_root)", 0.10, 2.00, 1.00, 0.05)
    pitch_axis = st.slider("Pitch axis (x/c)", 0.00, 0.50, 0.25, 0.05)

    st.subheader("Flight & Simulation")
    blades = st.number_input("Blade count b", min_value=1, max_value=12, value=2, step=1)
    rpm = st.slider("RPM", 400.0, 2500.0, 1200.0, 50.0)
    v_axial = st.slider("Axial velocity V_axial [m/s]", 0.0, 25.0, 0.0, 0.5)
    n_elem = st.slider("Radial strips N", 10, 80, 40, 5)

    st.subheader("3-D Display")
    view_mode = st.radio(
        "Mesh type",
        ["Full Airfoil (Lofted)", "Mean Camber Line"],
        horizontal=True,
    )

    st.subheader("Camera Preset")
    col_a, col_b = st.columns(2)
    if col_a.button("🔝 Top"):
        st.session_state["elev"], st.session_state["azim"] = 90, -90
    if col_b.button("👁️ Side"):
        st.session_state["elev"], st.session_state["azim"] = 0, -90
    col_c, col_d = st.columns(2)
    if col_c.button("📐 Iso"):
        st.session_state["elev"], st.session_state["azim"] = 20, -65
    if col_d.button("↔️ Front"):
        st.session_state["elev"], st.session_state["azim"] = 0, 0

# ── compute ───────────────────────────────────────────────────────────────────
valid_df = edited_df.dropna(subset=["r/R", "Airfoil"]).sort_values("r/R")
if len(valid_df) == 0:
    valid_df = pd.DataFrame({"r/R": [0.5], "Airfoil": ["NACA 0012"]})
    
r_stations = valid_df["r/R"].tolist()
af_names = valid_df["Airfoil"].tolist()

models = [_model(name, ncrit) for name in af_names]
af_blend = BlendedAirfoil(r_stations, models)

r_rc = min(0.125, radius * 0.2)

geom = RotorGeometry(
    radius=radius,
    root_cutout=r_rc,
    num_blades=blades,
    chord_func=make_chord_func(c_root, taper, radius, r_rc),
    twist_func=make_twist_func(pitch, twist, radius),
)
cond = FlightCondition(v_axial=v_axial, rpm=rpm, rho=1.225)
res = run_bemt(geom, cond, af_blend, num_elements=n_elem)

# ── telemetry HUD ─────────────────────────────────────────────────────────────
p_hp = res.power / 745.7
eff_label = (
    f"FM = **{res.figure_of_merit:.3f}**"
    if v_axial == 0.0
    else f"η_p = **{res.propulsive_eff * 100:.1f} %**"
)
stall_col = (
    "🔴" if res.stall_fraction > 0.05
    else ("🟡" if res.stall_fraction > 0 else "🟢")
)

hud_cols = st.columns(7)
hud_cols[0].metric("Thrust", f"{res.thrust:.1f} N")
hud_cols[1].metric("Power", f"{res.power:.0f} W  ({p_hp:.2f} hp)")
hud_cols[2].metric("Torque", f"{res.torque:.2f} N·m")
hud_cols[3].metric("C_T", f"{res.ct:.4f}")
hud_cols[4].metric("C_P", f"{res.cp:.5f}")
hud_cols[5].metric("M_tip", f"{res.tip_mach:.3f}")
hud_cols[6].metric(f"Stall {stall_col}", f"{res.stall_fraction * 100:.1f} %")

st.caption(f"Solidity σ = {res.solidity:.3f}  |  {eff_label}  |  C_Q = {res.cq:.5f}")

if af_blend.has_polars:
    st.success(f"Polars loaded for {len(af_names)} airfoil sections.", icon="✅")
else:
    st.warning("No polars found — using analytical fallback.", icon="⚠️")

st.markdown("---")

r_norm = res.r_stations / radius
deg_alpha = np.degrees(res.alpha)

tab_perf, tab_inspect = st.tabs(["🚁 Rotor Performance", "🔍 Station Inspector"])

with tab_perf:
    fig_2d = plt.figure(figsize=(6, 8))
    gs = fig_2d.add_gridspec(4, 1, hspace=0.48, left=0.15, right=0.95, top=0.95, bottom=0.06)

    ax_aoa = fig_2d.add_subplot(gs[0, 0])
    ax_thrust = fig_2d.add_subplot(gs[1, 0])
    ax_power = fig_2d.add_subplot(gs[2, 0])
    ax_inflow = fig_2d.add_subplot(gs[3, 0])

    _2d_axes = (ax_aoa, ax_thrust, ax_power, ax_inflow)
    for ax in _2d_axes:
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.35, color="gray")

    ax_aoa.plot(r_norm, deg_alpha, color="#4fc3f7", lw=1.8, label="α(r)")
    max_stall = max([np.degrees(m.alpha_stall) for m in af_blend.models])
    ax_aoa.axhline(max_stall, color="tomato", linestyle="--", lw=1.3, label="α_stall (max)")
    ax_aoa.set_ylabel("AoA [deg]", fontsize=7.5)
    ax_aoa.set_title(f"AoA Profile  (θ₀ = {pitch:.1f}°, twist = {twist:.1f}°)", fontsize=8)
    ax_aoa.set_xlim(r_norm[0], 1.0)
    ax_aoa.set_ylim(min(np.min(deg_alpha) - 2, -5), max(np.max(deg_alpha) + 2, 20))
    ax_aoa.legend(loc="upper right", fontsize=6.5)

    dt_vals = res.d_thrust * res.dr
    ax_thrust.plot(r_norm, dt_vals, color="#66bb6a", lw=1.8, label="dT [N]")
    ax_thrust.set_ylabel("dT [N]", fontsize=7.5)
    ax_thrust.set_title(f"Thrust Loading   T = {res.thrust:.1f} N", fontsize=8)
    ax_thrust.set_xlim(r_norm[0], 1.0)
    max_t = max(np.max(dt_vals), 0.05)
    ax_thrust.set_ylim(min(np.min(dt_vals), 0) * 1.2 or -0.05, max_t * 1.25)
    ax_thrust.legend(loc="upper right", fontsize=6.5)

    dp_vals = res.d_power * res.dr
    dq_vals = res.d_torque * res.dr
    ax_power.plot(r_norm, dp_vals, color="tomato", lw=1.8, label="dP [W]")
    ax_power.plot(r_norm, dq_vals * 10, color="#ce93d8", lw=1.5, linestyle="--", label="10×dQ [N·m]")
    ax_power.set_ylabel("Power / Torque", fontsize=7.5)
    ax_power.set_title(f"Power & Torque   P = {res.power:.0f} W  |  Q = {res.torque:.2f} N·m", fontsize=8)
    ax_power.set_xlim(r_norm[0], 1.0)
    max_p = max(np.max(dp_vals), np.max(dq_vals * 10), 0.1)
    ax_power.set_ylim(min(np.min(dp_vals), 0) * 1.2 or -0.05, max_p * 1.25)
    ax_power.legend(loc="upper right", fontsize=6.5)

    ax_inflow.plot(r_norm, res.inflow_ratio, color="black", lw=1.8, label="λ_total")
    ax_inflow.plot(r_norm, res.lambda_i, color="cyan", lw=1.5, linestyle="--", label="λ_i (induced)")
    ax_inflow.axhline(res.lambda_c, color="orange", lw=1.3, linestyle=":", label="λ_c (climb)")
    ax_inflow.set_xlabel("r / R", fontsize=7.5)
    ax_inflow.set_ylabel("Inflow λ", fontsize=7.5)
    ax_inflow.set_title("Inflow Ratio  λ = λ_c + λ_i", fontsize=8)
    ax_inflow.set_xlim(r_norm[0], 1.0)
    max_lam = max(np.max(res.inflow_ratio) * 1.2, 0.05)
    min_lam = min(np.min(res.inflow_ratio) * 1.2, 0.0)
    ax_inflow.set_ylim(min_lam, max_lam)
    ax_inflow.legend(loc="upper right", fontsize=6.5)

    # ── 3-D blade mesh (Plotly) ───────────────────────────────────────────────────
    r_edges_3d = np.linspace(r_rc, radius, n_elem + 1)
    use_camber = "Camber" in view_mode

    X_grid, Y_grid, Z_grid, C_grid = [], [], [], []

    for i in range(n_elem + 1):
        r_val = r_edges_3d[i]
        c_val = geom.chord_func(r_val)
        th_val = geom.twist_func(r_val)
        
        af_x, af_y = af_blend.get_blended_coords(r_val / radius, use_camber=use_camber)
        
        # Downsample for faster 3D rendering performance
        step = max(1, len(af_x) // 50)
        af_x, af_y = af_x[::step], af_y[::step]
        
        xr = (pitch_axis - af_x) * c_val * np.cos(th_val) - af_y * c_val * np.sin(th_val)
        zr = (pitch_axis - af_x) * c_val * np.sin(th_val) + af_y * c_val * np.cos(th_val)
        
        elem_idx = min(i, n_elem - 1)
        is_stalled = res.stalled_mask[elem_idx]
        
        X_grid.append(xr)
        Y_grid.append(np.full_like(xr, r_val))
        Z_grid.append(zr)
        C_grid.append(np.full_like(xr, 1.0 if is_stalled else 0.0))

    X_grid = np.array(X_grid)
    Y_grid = np.array(Y_grid)
    Z_grid = np.array(Z_grid)
    C_grid = np.array(C_grid)

    title_mode = "Single Blade 3-D Reference"
    af_desc = " → ".join(af_names)

    fig_3d = go.Figure(data=[go.Surface(
        x=Y_grid, y=X_grid, z=Z_grid, 
        surfacecolor=C_grid,
        colorscale=[[0, 'rgb(38,178,228)'], [1, 'rgb(255,64,64)']],
        showscale=False,
        cmin=0, cmax=1,
        lighting=dict(ambient=0.7, diffuse=0.9, roughness=1.0, specular=0.0, fresnel=0.0)
    )])

    max_c = max(c_root, c_root * taper)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=2.0 * np.cos(np.radians(st.session_state["azim"])) * np.cos(np.radians(st.session_state["elev"])),
            y=2.0 * np.sin(np.radians(st.session_state["azim"])) * np.cos(np.radians(st.session_state["elev"])),
            z=2.0 * np.sin(np.radians(st.session_state["elev"]))
        )
    )

    fig_3d.update_layout(
        title=f"{title_mode}<br><sup>{af_desc}</sup>",
        scene=dict(
            xaxis_title='Radius r [m]',
            yaxis_title='Advancing x [m]',
            zaxis_title='Height z [m]',
            xaxis=dict(range=[0, radius + 0.04]),
            yaxis=dict(range=[-max_c, max_c]),
            aspectmode='manual',
            aspectratio=dict(x=3.0, y=1.0, z=0.15),
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        height=500
    )

    # --------------------------------------------------------------------------
    # 2D TOP VIEW INDICATOR
    # --------------------------------------------------------------------------
    fig_top, ax_top = plt.subplots(figsize=(3, 3))
    ax_top.add_patch(plt.Circle((0, 0), r_rc, color='gray', zorder=5))
    ax_top.add_patch(plt.Circle((0, 0), radius, color='lightblue', alpha=0.2, zorder=1))
    
    for k in range(blades):
        psi = k * (2 * np.pi / blades)
        # Approximate blade outline
        x0, y0 = r_rc * np.cos(psi), r_rc * np.sin(psi)
        x1, y1 = radius * np.cos(psi), radius * np.sin(psi)
        dx = c_root * 0.4 * np.sin(psi)
        dy = -c_root * 0.4 * np.cos(psi)
        
        blade_poly = [
            [x0 - dx, y0 - dy],
            [x0 + dx, y0 + dy],
            [x1 + dx, y1 + dy],
            [x1 - dx, y1 - dy]
        ]
        ax_top.add_patch(plt.Polygon(blade_poly, color='#4fc3f7', ec='black', lw=0.5, zorder=3))

    ax_top.set_aspect('equal')
    ax_top.axis('off')
    ax_top.set_xlim(-radius * 1.1, radius * 1.1)
    ax_top.set_ylim(-radius * 1.1, radius * 1.1)
    ax_top.set_title(f"Rotor Top View ({blades} Blades)", fontsize=10, pad=12)
    fig_top.tight_layout()

    # --------------------------------------------------------------------------
    # RENDER COLUMNS
    # --------------------------------------------------------------------------
    col_3d, col_2d = st.columns([1.5, 1.0])
    with col_3d:
        # Display the 2D Top View and 3D reference blade
        t1, t2 = st.columns([1, 2])
        with t1:
            st.pyplot(fig_top, use_container_width=True)
            plt.close(fig_top)
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_2d:
        st.pyplot(fig_2d, use_container_width=True)
        plt.close(fig_2d)

@st.fragment
def render_station_inspector(af_blend, res, geom, cond, r_norm, deg_alpha, r_rc, radius):
    st.subheader("Spanwise Station Inspector")
    min_rnorm = float(r_rc / radius)
    
    if "inspect_r" not in st.session_state or st.session_state.inspect_r < min_rnorm or st.session_state.inspect_r > 1.0:
        st.session_state.inspect_r = min_rnorm + (1.0 - min_rnorm)/2
        
    def update_from_slider():
        st.session_state.inspect_r = st.session_state._sl_inspect
    def update_from_num():
        st.session_state.inspect_r = st.session_state._num_inspect

    col_sl, col_num = st.columns([3, 1])
    with col_sl:
        st.slider("Select radial station (r/R)", min_value=min_rnorm, max_value=1.0, 
                  value=st.session_state.inspect_r, step=0.01, key="_sl_inspect", on_change=update_from_slider)
    with col_num:
        st.number_input("Exact input (r/R)", min_value=min_rnorm, max_value=1.0, 
                        value=st.session_state.inspect_r, step=0.01, format="%.4f", key="_num_inspect", on_change=update_from_num)
        
    inspect_r = st.session_state.inspect_r
    
    idx_r = (np.abs(r_norm - inspect_r)).argmin()
    local_alpha = deg_alpha[idx_r]
    local_cl = res.cl[idx_r]
    local_cd = res.cd[idx_r]
    local_chord = geom.chord_func(inspect_r * radius)
    local_twist = np.degrees(geom.twist_func(inspect_r * radius))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chord", f"{local_chord:.4f} m")
    col2.metric("Twist", f"{local_twist:.1f}°")
    col3.metric("Operating AoA", f"{local_alpha:.2f}°")
    col4.metric("Lift Coeff (Cl)", f"{local_cl:.4f}")
    col5.metric("Drag Coeff (Cd)", f"{local_cd:.5f}")
    
    blend_x, blend_y = af_blend.get_blended_coords(inspect_r, use_camber=False)
    fig_shape, ax_shape = plt.subplots(figsize=(10, 2))
    ax_shape.plot(blend_x, blend_y, "k-", lw=1.5)
    ax_shape.fill(blend_x, blend_y, color="gray", alpha=0.15)
    ax_shape.set_aspect("equal")
    ax_shape.axis("off")
    ax_shape.set_title(f"Blended Airfoil Profile at r/R = {inspect_r:.2f}", fontsize=10)
    st.pyplot(fig_shape, use_container_width=True)
    plt.close(fig_shape)
    
    st.markdown("---")
    
    local_v_t = cond.omega * (inspect_r * radius)
    local_v_p = res.inflow_ratio[idx_r] * (cond.omega * radius)
    local_u_res = np.hypot(local_v_t, local_v_p)
    T_air = 288.15
    mu_air = 1.458e-6 * T_air ** 1.5 / (T_air + 110.4)
    local_re = (cond.rho * local_u_res * local_chord) / mu_air
    local_mach = local_v_t / cond.speed_of_sound
    
    st.markdown(f"**Aerodynamic Response at r/R = {inspect_r:.2f}**  (Interpolated for local Re = {int(local_re):,}, Mach = {local_mach:.3f})")
    
    alpha_sweep = np.linspace(-10, 15, 150)
    a_rad = np.radians(alpha_sweep)
    r_n_arr = np.full_like(a_rad, inspect_r)
    re_arr = np.full_like(a_rad, local_re)
    m_arr = np.full_like(a_rad, local_mach)
    
    cl_sweep, cd_sweep, _ = af_blend.evaluate_vectorized(a_rad, re_arr, m_arr, r_n_arr)
    
    fig_polar, (ax_l, ax_d, ax_p) = plt.subplots(1, 3, figsize=(14, 4))
    for ax in (ax_l, ax_d, ax_p):
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.35, color="gray")
        
    ax_l.plot(alpha_sweep, cl_sweep, color="#1f77b4", lw=2)
    ax_l.axvline(local_alpha, color="tomato", linestyle="--", lw=1.5, label=f"Operating AoA ({local_alpha:.1f}°)")
    ax_l.set_xlabel("AoA [deg]", fontsize=8)
    ax_l.set_ylabel("Lift Coefficient Cl", fontsize=8)
    ax_l.set_title("Lift Polar", fontsize=9)
    ax_l.legend(fontsize=7)
    
    ax_d.plot(alpha_sweep, cd_sweep, color="#d62728", lw=2)
    ax_d.axvline(local_alpha, color="tomato", linestyle="--", lw=1.5, label=f"Operating AoA ({local_alpha:.1f}°)")
    ax_d.set_xlabel("AoA [deg]", fontsize=8)
    ax_d.set_ylabel("Drag Coefficient Cd", fontsize=8)
    ax_d.set_title("Drag Polar", fontsize=9)
    ax_d.legend(fontsize=7)
    
    ax_p.plot(cd_sweep, cl_sweep, color="#2ca02c", lw=2)
    ax_p.plot([local_cd], [local_cl], marker="o", color="tomato", markersize=6, label=f"Operating Point")
    ax_p.set_xlabel("Drag Coefficient Cd", fontsize=8)
    ax_p.set_ylabel("Lift Coefficient Cl", fontsize=8)
    ax_p.set_title("Drag Polar (Cl vs Cd)", fontsize=9)
    ax_p.legend(fontsize=7)
    
    st.pyplot(fig_polar, use_container_width=True)
    plt.close(fig_polar)

with tab_inspect:
    render_station_inspector(af_blend, res, geom, cond, r_norm, deg_alpha, r_rc, radius)

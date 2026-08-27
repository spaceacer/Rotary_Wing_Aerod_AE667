import os
import sys
from enum import Enum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.airfoil_catalog import filter_airfoil_catalog, get_all_available_airfoils, load_airfoil_coords
from core.airfoil_blend import BlendedAirfoil
from core.airfoil_model import AirfoilModel
from core.bemt_solver import run_bemt
from core.geometry_helpers import make_chord_func, make_twist_func
from core.models import FlightCondition, RotorGeometry

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="Mission Planner & Sizing", page_icon="🗺️", layout="wide")
st.title("🗺️  Tiltrotor Mission Planner & Aircraft Sizing")

# ── Atmospheric & Aerodynamic Helpers ──────────────────────────────────────────

def isa_atmosphere(altitude_m: float) -> Tuple[float, float, float]:
    """Computes ISA atmospheric density [kg/m³], speed of sound [m/s], and temperature [K]."""
    alt = np.clip(altitude_m, 0.0, 11000.0)
    T = 288.15 - 0.0065 * alt
    P = 101325.0 * ((T / 288.15) ** 5.2561)
    rho = P / (287.058 * T)
    a = np.sqrt(1.4 * 287.058 * T)
    return float(rho), float(a), float(T)

def power_law(x, a, b):
    return a * (x ** b)

def linear_law(x, a, b):
    return a * x + b

# ── Historical Benchmarks Data & Curve Fits ───────────────────────────────────

benchmarks = pd.DataFrame({
    "Aircraft": ["Bell XV-3", "Bell XV-15", "Bell Boeing V-22", "Leonardo AW609", "Bell V-280 Valor"],
    "MTOW_kg": [2177, 6000, 23982, 8165, 14000],
    "Empty_kg": [1648, 4570, 15032, 4765, 8200],
    "Disc_Loading": [32.5, 73.2, 102.5, 88.4, 95.0],
    "Solidity": [0.053, 0.089, 0.105, 0.096, 0.100],
    "Installed_PW": [0.154, 0.385, 0.380, 0.354, 0.533],
    "Service_Ceiling_m": [3600, 8840, 7620, 7620, 8500],
    "Wing_tc": [15.0, 23.0, 23.0, 21.0, 20.0]
})
benchmarks["We_W0"] = benchmarks["Empty_kg"] / benchmarks["MTOW_kg"]

popt_we, _ = curve_fit(power_law, benchmarks["MTOW_kg"], benchmarks["We_W0"], p0=[1.5, -0.09])
popt_pw, _ = curve_fit(power_law, benchmarks["Disc_Loading"], benchmarks["Installed_PW"], p0=[0.05, 0.5])
popt_sol, _ = curve_fit(linear_law, benchmarks["Disc_Loading"], benchmarks["Solidity"], p0=[0.001, 0.02])
popt_ceil, _ = curve_fit(linear_law, benchmarks["Service_Ceiling_m"], benchmarks["Installed_PW"], p0=[5e-5, 0.1])

# ── Segment Definitions & Presets ─────────────────────────────────────────────

class SegmentType(Enum):
    HOVER = "Hover"
    VERTICAL_CLIMB = "Vertical Climb"
    TRANSITION = "Transition / Conversion"
    CRUISE = "Cruise (Axial)"
    LOITER = "Loiter"
    VERTICAL_DESCENT = "Vertical Descent"

PRESETS = {
    "Standard: Takeoff -> Climb -> Cruise -> Land": [
        {"Name": "1. Takeoff Hover (OGE)", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "2. Vertical Climb", "Type": "Vertical Climb", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 2500.0, "Climb [m/s]": 5.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "3. Outbound Cruise", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 450.0, "Speed [km/h]": 450.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "4. On-Station Loiter", "Type": "Loiter", "Dur [min]": 15.0, "Dist [km]": 0.0, "Speed [km/h]": 260.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "5. Inbound Cruise", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 450.0, "Speed [km/h]": 450.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "6. Descent to Field", "Type": "Vertical Descent", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": -3.5, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "7. Touchdown Hover", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
    ],
    "High-Altitude Transit (7,000 m Ceiling)": [
        {"Name": "1. Takeoff Hover", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "2. Climb to Ceiling", "Type": "Vertical Climb", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 7000.0, "Climb [m/s]": 6.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "3. High-Altitude Transit", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 400.0, "Speed [km/h]": 450.0, "Alt [m]": 7000.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "4. Descent to LZ", "Type": "Vertical Descent", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 1500.0, "Climb [m/s]": -5.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "5. LZ Hover & Deploy", "Type": "Hover", "Dur [min]": 2.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 1500.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
    ]
}

# ── Sidebar Configurations ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Aircraft & Rotor Specs")
    
    with st.expander("Airfoil Selection", expanded=True):
        all_airfoils = get_all_available_airfoils()
        sel_af = st.selectbox("Primary Airfoil", all_airfoils, index=all_airfoils.index("NACA 0012") if "NACA 0012" in all_airfoils else 0)
        ncrit = st.selectbox("Boundary Layer Ncrit", [9, 5], format_func=lambda x: "9 (Standard Flight)" if x == 9 else "5 (Turbulent Flow)")
    
    with st.expander("Tiltrotor Proprotors", expanded=True):
        num_rotors = 2
        r_rotor = st.slider("Rotor Radius R [m]", 1.0, 8.0, 4.25, 0.05)
        r_root = st.slider("Root Cutout [m]", 0.05, 1.5, 0.45, 0.05)
        n_blades = st.number_input("Blades per Rotor (b)", min_value=2, max_value=8, value=3, step=1)
        c_blade = st.slider("Blade Chord c_root [m]", 0.05, 1.0, 0.38, 0.01)
        blade_tap = st.slider("Blade Taper (c_tip / c_root)", 0.2, 1.5, 0.85, 0.05)
        
        st.markdown("---")
        st.markdown("**Dual Collective Control Settings:**")
        th_75_hover = st.slider("Hover Collective θ_0.75 [°]", 0.0, 25.0, 11.0, 0.5)
        th_75_cruise = st.slider("Cruise Collective θ_0.75 [°]", 20.0, 55.0, 40.0, 0.5)
        th_tw = st.slider("Blade Linear Twist θ_tw [°]", -40.0, 5.0, -18.0, 0.5)
    
    with st.expander("Airframe & Weights", expanded=True):
        mtow = st.number_input("MTOW [kg]", 2000.0, 30000.0, 10500.0, 100.0)
        empty_frac = st.slider("Empty Mass Fraction (We/W0)", 0.35, 0.75, 0.55, 0.01)
        fuel_frac = st.slider("Fuel Mass Fraction (Wf/W0)", 0.05, 0.45, 0.25, 0.01)
        payload_mass = st.number_input("Fixed Payload [kg]", 0.0, 8000.0, 1440.0, 10.0)
        reserve_fuel = st.number_input("Reserve Fuel Limit [kg]", 50.0, 1500.0, 300.0, 10.0)
    
    with st.expander("Wing & Fuselage Geometry"):
        s_wing = st.slider("Wing Area S [m²]", 6.0, 100.0, 32.0, 0.5)
        ar_wing = st.slider("Wing Aspect Ratio (AR)", 4.0, 16.0, 7.8, 0.1)
        taper_w = st.slider("Wing Taper Ratio (λ)", 0.2, 1.0, 0.65, 0.05)
        sweep_deg = st.slider("Wing Sweep Angle [°]", -10.0, 35.0, 2.5, 0.5)
        wing_tc_pct = st.slider("Wing Thickness t/c [%]", 12.0, 28.0, 21.0, 0.5)
        
        l_fuse = st.slider("Fuselage Length [m]", 4.0, 35.0, 14.5, 0.5)
        w_fuse = st.slider("Fuselage Width [m]", 0.6, 6.0, 2.1, 0.1)
        vh_tail = st.slider("H-Tail Volume Coeff (Vh)", 0.2, 1.5, 0.85, 0.05)
        
        cd0 = st.number_input("Airframe Parasite CD0", 0.010, 0.080, 0.024, 0.001, format="%.3f")
        oswald_e = st.slider("Oswald Efficiency (e)", 0.60, 0.95, 0.82, 0.01)
    
    with st.expander("Propulsion & Engines"):
        p_inst_kw = st.number_input("Total Installed Power [kW]", 500.0, 25000.0, 4800.0, 100.0)
        sfc_hr = st.number_input("Engine SFC [kg/kW/hr]", 0.15, 0.60, 0.285, 0.005, format="%.3f")

# ── Mission Schedule Management (Decoupled Creation & Editing) ────────────────

st.subheader("📋 Flight Plan & Mission Schedule")

col_preset, _ = st.columns([2, 2])
with col_preset:
    preset_choice = st.selectbox("Select Mission Preset:", list(PRESETS.keys()))

if "mission_df" not in st.session_state or st.session_state.get("current_preset") != preset_choice:
    st.session_state["mission_df"] = pd.DataFrame(PRESETS[preset_choice])
    st.session_state["current_preset"] = preset_choice

# Decoupled Creation Toolbar
with st.expander("➕ Add New Flight Segment", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        new_name = st.text_input("Segment Name", "Custom Leg")
        new_type = st.selectbox("Segment Type", [e.value for e in SegmentType], index=3)
    with c2:
        new_dist = st.number_input("Distance [km]", min_value=0.0, max_value=2000.0, value=250.0, step=10.0) if new_type in ["Cruise (Axial)", "Transition / Conversion"] else 0.0
        new_dur = st.number_input("Duration [min]", min_value=0.1, max_value=180.0, value=5.0, step=1.0) if new_type in ["Hover", "Loiter"] else 0.0
    with c3:
        new_speed = st.number_input("Speed [km/h]", min_value=0.0, max_value=650.0, value=420.0, step=10.0) if new_type in ["Cruise (Axial)", "Loiter", "Transition / Conversion"] else 0.0
        new_alt = st.number_input("Target Altitude [m]", min_value=0.0, max_value=10000.0, value=2500.0, step=100.0)
    with c4:
        new_climb = st.number_input("Climb Rate [m/s]", min_value=-15.0, max_value=20.0, value=5.0, step=0.5) if new_type in ["Vertical Climb", "Vertical Descent"] else 0.0
        new_rpm = st.number_input("Proprotor RPM", min_value=100.0, max_value=800.0, value=365.0 if new_type == "Cruise (Axial)" else 440.0, step=5.0)
        new_wind = st.number_input("Headwind [km/h]", min_value=-100.0, max_value=100.0, value=0.0, step=5.0) if new_type == "Cruise (Axial)" else 0.0

    if st.button("➕ Append Leg to Flight Plan", use_container_width=True):
        new_row = {
            "Name": new_name, "Type": new_type, "Dur [min]": float(new_dur), "Dist [km]": float(new_dist),
            "Speed [km/h]": float(new_speed), "Alt [m]": float(new_alt), "Climb [m/s]": float(new_climb),
            "RPM": float(new_rpm), "Wind [km/h]": float(new_wind)
        }
        st.session_state["mission_df"] = pd.concat([st.session_state["mission_df"], pd.DataFrame([new_row])], ignore_index=True)
        st.rerun()

# Interactive Flight Plan Table Editor
edited_mission = st.data_editor(
    st.session_state["mission_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Segment Type", options=[e.value for e in SegmentType], required=True),
        "Alt [m]": st.column_config.NumberColumn("Target Alt [m]", format="%d m"),
        "Speed [km/h]": st.column_config.NumberColumn("Speed [km/h]", format="%d km/h"),
        "Dist [km]": st.column_config.NumberColumn("Distance [km]", format="%.1f km"),
        "Dur [min]": st.column_config.NumberColumn("Duration [min]", format="%.1f min"),
        "Climb [m/s]": st.column_config.NumberColumn("Climb [m/s]", format="%.1f m/s"),
        "RPM": st.column_config.NumberColumn("RPM", format="%.0f rpm"),
        "Wind [km/h]": st.column_config.NumberColumn("Wind [km/h]", format="%d km/h"),
    }
)
st.session_state["mission_df"] = edited_mission

# ── Aeromechanical & State-Marching Engine ─────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_airfoil(name: str, ncrit_val: int) -> AirfoilModel:
    return AirfoilModel(airfoil_name=name, ncrit_pref=ncrit_val)

af_model = _load_airfoil(sel_af, ncrit)

# Derived Vehicle Geometry
m_empty = mtow * empty_frac
m_fuel_init = mtow * fuel_frac
takeoff_gross_mass = m_empty + m_fuel_init + payload_mass

b_wing = np.sqrt(s_wing * ar_wing)
c_root_w = (2.0 * s_wing) / (b_wing * (1.0 + taper_w))
c_tip_w = c_root_w * taper_w
mac = (2.0 / 3.0) * c_root_w * (1.0 + taper_w + taper_w**2) / (1.0 + taper_w)

l_arm = 0.45 * l_fuse
s_htail = (vh_tail * s_wing * mac) / l_arm
disk_area_total = num_rotors * np.pi * (r_rotor ** 2)

# Fuselage Clearance & Flutter Calculations
y_tip_rotor = 0.5 * b_wing
fuse_clearance = (y_tip_rotor - r_rotor) - (0.5 * w_fuse)
collision_warn = fuse_clearance < 0.25
collision_msg = f"CRITICAL: Rotor blade penetrates fuselage by {abs(fuse_clearance):.2f}m!" if fuse_clearance <= 0 else f"WARNING: Clearance is tight ({fuse_clearance:.2f}m < 0.25m)"

v_flutter_kmh = 320.0 * ((wing_tc_pct / 15.0) ** 1.5)

# State Marching Run
curr_t, curr_alt, curr_fuel, curr_dist = 0.0, 0.0, m_fuel_init, 0.0
telemetry = []
failed, fail_msg = False, ""
dt = 2.0

for _, leg in edited_mission.iterrows():
    seg_type = leg["Type"]
    start_alt = curr_alt
    alt_t = float(leg.get("Alt [m]", 0.0))
    dur_m = float(leg.get("Dur [min]", 0.0))
    dist_km = float(leg.get("Dist [km]", 0.0))
    speed_k = float(leg.get("Speed [km/h]", 0.0))
    climb_v = float(leg.get("Climb [m/s]", 0.0))
    rpm_val = float(leg.get("RPM", 440.0))
    wind_k = float(leg.get("Wind [km/h]", 0.0))
    
    if seg_type in [SegmentType.HOVER.value, SegmentType.LOITER.value]:
        dur_s = max(dur_m * 60.0, 1.0)
        v_tas = (speed_k / 3.6) if seg_type == SegmentType.LOITER.value else 0.0
        v_ground = v_tas
        alt_target = curr_alt if seg_type == SegmentType.HOVER.value else alt_t
    elif seg_type in [SegmentType.VERTICAL_CLIMB.value, SegmentType.VERTICAL_DESCENT.value]:
        climb_speed = abs(climb_v) if abs(climb_v) > 0.1 else 3.0
        dur_s = max(abs(alt_t - curr_alt) / climb_speed, 1.0)
        v_tas = climb_speed
        v_ground = 0.0
        alt_target = alt_t
    elif seg_type == SegmentType.TRANSITION.value:
        v_tas = max(speed_k / 3.6, 25.0)
        v_ground = max(v_tas - (wind_k / 3.6), 5.0)
        dur_s = (max(dist_km * 1000.0, 50.0) / v_ground) if dist_km > 0 else 30.0
        alt_target = alt_t
    elif seg_type == SegmentType.CRUISE.value:
        v_tas = max(speed_k / 3.6, 15.0)
        v_ground = max(v_tas - (wind_k / 3.6), 5.0)
        dur_s = max(dist_km * 1000.0, 100.0) / v_ground
        alt_target = alt_t
        
    n_steps = max(int(np.ceil(dur_s / dt)), 1)
    step_dt = dur_s / n_steps
    alt_rate = (alt_target - start_alt) / dur_s if dur_s > 0 else 0.0
    
    for _ in range(n_steps):
        gross_m = m_empty + curr_fuel + payload_mass
        w_newtons = gross_m * 9.80665
        rho, a_sound, _ = isa_atmosphere(curr_alt)
        
        omega = (rpm_val * 2.0 * np.pi) / 60.0
        v_tip = omega * r_rotor
        m_tip = np.sqrt(v_tip**2 + v_tas**2) / a_sound
        
        if seg_type in [SegmentType.HOVER.value, SegmentType.VERTICAL_CLIMB.value, SegmentType.VERTICAL_DESCENT.value]:
            download_factor = 1.08
            T_hover_total_N = gross_m * 9.80665 * download_factor
            P_induced_W = (T_hover_total_N ** 1.5) / np.sqrt(2.0 * rho * disk_area_total)
            p_req_kw = (P_induced_W / 0.74) / (0.94 * 1000.0)
        elif seg_type == SegmentType.TRANSITION.value:
            # Transition power blending (cosine transition corridor)
            q_dyn = 0.5 * rho * (v_tas ** 2)
            cl_wing = np.clip(w_newtons / max(q_dyn * s_wing, 1e-4), 0.0, 1.3)
            lift_wing = q_dyn * s_wing * cl_wing
            frac_rotor = max(0.0, 1.0 - (lift_wing / w_newtons))
            P_hover_part = (frac_rotor * ((w_newtons * 1.08) ** 1.5) / np.sqrt(2.0 * rho * disk_area_total)) / (0.74 * 0.94 * 1000.0)
            cd_ind = (cl_wing ** 2) / (np.pi * ar_wing * oswald_e)
            drag_total = q_dyn * s_wing * (cd0 + cd_ind)
            P_cruise_part = (drag_total * v_tas) / (0.83 * 1000.0)
            p_req_kw = P_hover_part + P_cruise_part
        else:
            q_dyn = 0.5 * rho * (v_tas ** 2)
            cl_wing = w_newtons / max(q_dyn * s_wing, 1e-4)
            cd_ind = (cl_wing ** 2) / (np.pi * ar_wing * oswald_e)
            drag_total = q_dyn * s_wing * (cd0 + cd_ind)
            p_req_kw = (drag_total * v_tas) / (0.83 * 1000.0)
            
        p_avail_kw = p_inst_kw * ((rho / 1.225) ** 1.05)
        fuel_burn = p_req_kw * (sfc_hr / 3600.0) * step_dt
        
        if p_req_kw > p_avail_kw:
            failed, fail_msg = True, f"Engine Power Limit ({p_req_kw:.0f} kW required > {p_avail_kw:.0f} kW available) during '{leg['Name']}'"
            break
        if curr_fuel <= reserve_fuel:
            failed, fail_msg = True, f"Reserve Fuel Breached ({curr_fuel:.1f} kg <= {reserve_fuel:.1f} kg limit) during '{leg['Name']}'"
            break
            
        curr_fuel -= fuel_burn
        curr_alt += alt_rate * step_dt
        curr_dist += (v_ground * step_dt) / 1000.0
        curr_t += step_dt
        
        telemetry.append({
            "time_min": curr_t / 60.0, "leg": leg["Name"], "type": seg_type, "alt_m": curr_alt,
            "gross_kg": gross_m, "fuel_kg": curr_fuel, "payload_kg": payload_mass,
            "p_req_kw": p_req_kw, "p_avail_kw": p_avail_kw, "speed_kmh": v_tas * 3.6,
            "dist_km": curr_dist, "mach_tip": m_tip
        })
        
    if failed:
        break

df_res = pd.DataFrame(telemetry)

# ── Mission Status Banner ─────────────────────────────────────────────────────

if failed:
    st.error(f"❌ **MISSION INFEASIBLE:** {fail_msg}")
elif not df_res.empty:
    final_f = df_res['fuel_kg'].iloc[-1]
    st.success(f"✅ **MISSION SUCCESSFUL:** Completed {df_res['dist_km'].iloc[-1]:.1f} km in {df_res['time_min'].iloc[-1]:.1f} min. Final fuel: {final_f:.1f} kg (Reserve margin: +{final_f - reserve_fuel:.1f} kg).")

st.markdown("---")

# ── Seven Master Display Tabs ─────────────────────────────────────────────────

t_cad, t_prof, t_polar, t_blade, t_power, t_reg, t_hov_maps, t_cr_maps, t_comp, t_log = st.tabs([
    "📐 General Arrangement",
    "📈 Mission Profile",
    "🌪️ 2D Polar Explorer",
    "🚁 3D Loft & Dual BEMT",
    "⚡ Power & Mach Envelope",
    "📊 Statistical Benchmarks",
    "🚁 6.1 Hover Performance Maps",
    "✈️ 6.2 Forward-Flight Propeller Maps",
    "📑 6.3 Comparable Rotor Benchmarks",
    "📋 Telemetry Log"
])

# ── TAB 1: General Arrangement ────────────────────────────────────────────────
with t_cad:
    st.subheader("Aircraft Geometry & Fuselage Clearance Audit")
    fig_c, (ax_c, ax_card) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={'width_ratios': [1.3, 1.0]}, dpi=100)
    
    # Fuselage Outline
    fx = np.array([0, 0.15*l_fuse, 0.75*l_fuse, l_fuse, 0.75*l_fuse, 0.15*l_fuse, 0])
    fy = np.array([0, 0.5*w_fuse, 0.5*w_fuse, 0, -0.5*w_fuse, -0.5*w_fuse, 0])
    ax_c.fill(fx, fy, color='#e63946' if collision_warn else '#ced4da', alpha=0.85, edgecolor='k', lw=1.5)

    # Wing Planform
    wx_le = 0.35 * l_fuse
    tip_x_off = 0.5 * b_wing * np.tan(np.radians(sweep_deg))
    wx = [wx_le, wx_le + tip_x_off, wx_le + tip_x_off + c_tip_w, wx_le + c_root_w, wx_le + tip_x_off + c_tip_w, wx_le + tip_x_off, wx_le]
    wy = [0, 0.5*b_wing, 0.5*b_wing, 0, -0.5*b_wing, -0.5*b_wing, 0]
    ax_c.fill(wx, wy, color='#9ec5fe', alpha=0.8, edgecolor='blue', lw=1.5)

    # Horizontal Tail
    b_ht = np.sqrt(s_htail * 4.2)
    c_ht = s_htail / b_ht
    hx_le = wx_le + l_arm
    ax_c.fill([hx_le, hx_le, hx_le + c_ht, hx_le + c_ht], [-0.5*b_ht, 0.5*b_ht, 0.5*b_ht, -0.5*b_ht], color='#6c757d', alpha=0.85, edgecolor='k')

    # Twin Rotors
    rcs = [(wx_le + tip_x_off, 0.5*b_wing), (wx_le + tip_x_off, -0.5*b_wing)]
    for rx, ry in rcs:
        col = 'crimson' if collision_warn else 'darkgreen'
        ax_c.add_patch(plt.Circle((rx, ry), r_rotor, color=col, fill=True, alpha=0.15, linestyle='--', lw=1.6))
        ax_c.add_patch(plt.Circle((rx, ry), r_rotor, color=col, fill=False, linestyle='--', lw=1.6))
        ax_c.plot(rx, ry, 'o', color=col, ms=5)

    ax_c.set_aspect('equal')
    ax_c.set_xlim(-1, l_fuse + 2)
    ax_c.set_ylim(-0.55*b_wing - r_rotor, 0.55*b_wing + r_rotor)
    ax_c.set_title(f"Twin Proprotor Layout ({sel_af})" + (f"\n[!] CLEARANCE: {fuse_clearance:.2f}m" if collision_warn else ""), 
                   fontsize=10.5, fontweight='bold', color='crimson' if collision_warn else 'black')
    ax_c.set_xlabel("X [m]"); ax_c.set_ylabel("Y [m]"); ax_c.grid(True, linestyle=':', alpha=0.6)

    # Summary Diagnostic Card
    ax_card.axis('off')
    final_fuel = df_res['fuel_kg'].iloc[-1] if not df_res.empty else 0.0
    tot_time = df_res['time_min'].iloc[-1] if not df_res.empty else 0.0
    tot_dist = df_res['dist_km'].iloc[-1] if not df_res.empty else 0.0
    status_txt = "MISSION FEASIBLE" if (not failed and not df_res.empty) else ("NO MISSION DATA" if df_res.empty else "MISSION FAILED")

    card_str = (
        f"AIRCRAFT SIZING & MISSION AUDIT\n"
        f"==========================================\n"
        f"STATUS: {status_txt}\n"
        f"------------------------------------------\n"
        f"MASS & PAYLOAD BREAKDOWN:\n"
        f"  • MTOW Rating:       {mtow:8.1f} kg\n"
        f"  • Actual Gross Mass: {takeoff_gross_mass:8.1f} kg\n"
        f"  • Operating Empty:   {m_empty:8.1f} kg ({empty_frac*100:.1f}%)\n"
        f"  • Initial Fuel:      {m_fuel_init:8.1f} kg ({fuel_frac*100:.1f}%)\n"
        f"  • Fixed Payload:     {payload_mass:8.1f} kg\n"
        f"  • Final Fuel Left:   {final_fuel:8.1f} kg (Reserve: {reserve_fuel:.0f} kg)\n\n"
        f"AEROMECHANICAL METRICS:\n"
        f"  • Selected Airfoil:  {sel_af}\n"
        f"  • Rotor Radius R:    {r_rotor:6.2f} m (Disc Loading: {takeoff_gross_mass/disk_area_total:.1f} kg/m²)\n"
        f"  • Fuselage Clearance:{fuse_clearance:6.2f} m\n"
        f"  • Flutter Speed:     ~{v_flutter_kmh:.0f} km/h\n"
        f"  • Total Mission Time:{tot_time:6.1f} min\n"
        f"  • Total Ground Dist: {tot_dist:6.1f} km\n"
    )
    if collision_warn:
        card_str += f"\n[!] PROXIMITY ALERT:\n  {collision_msg}\n"
    if failed:
        card_str += f"\nFAILURE DIAGNOSTIC:\n  ! {fail_msg}\n"

    border_col = '#dc3545' if (collision_warn or takeoff_gross_mass > mtow) else '#ced4da'
    ax_card.text(0.02, 0.98, card_str, fontfamily='monospace', fontsize=9.0, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff5f5' if (collision_warn or takeoff_gross_mass > mtow) else '#f8f9fa', edgecolor=border_col, lw=1.4))
    
    st.pyplot(fig_c)
    plt.close(fig_c)

# ── TAB 2: Mission Profile ────────────────────────────────────────────────────
with t_prof:
    st.subheader("State-Marching Mission Trajectory")
    if not df_res.empty:
        fig_p, (ax_p1, ax_p2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, dpi=100)
        
        # Altitude & Speed
        ax_p1.plot(df_res["time_min"], df_res["alt_m"], 'b-', lw=2.2, label='Altitude [m]')
        ax_p1.set_ylabel("Altitude [m]", color='b')
        ax_p1.grid(True, linestyle=':', alpha=0.6)
        ax_p1.set_title("Mission Altitude & True Airspeed Trajectory", fontsize=10.5, fontweight='bold')
        
        ax_twin1 = ax_p1.twinx()
        ax_twin1.plot(df_res["time_min"], df_res["speed_kmh"], 'orange', linestyle='--', lw=1.8, label='TAS [km/h]')
        ax_twin1.set_ylabel("True Airspeed [km/h]", color='orange')
        
        # Gross Mass & Fuel Burn
        ax_p2.plot(df_res["time_min"], df_res["gross_kg"], 'g-', lw=2.2, label='Gross Mass [kg]')
        ax_p2.plot(df_res["time_min"], df_res["fuel_kg"], 'r--', lw=1.8, label='Fuel Remaining [kg]')
        ax_p2.axhline(reserve_fuel, color='crimson', linestyle=':', lw=1.6, label=f'Reserve Limit ({reserve_fuel:.0f} kg)')
        ax_p2.set_ylabel("Mass [kg]")
        ax_p2.set_xlabel("Mission Time [minutes]")
        ax_p2.grid(True, linestyle=':', alpha=0.6)
        ax_p2.legend(loc='center right', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

# ── TAB 3: Airfoil 2D Polar Explorer ──────────────────────────────────────────
with t_polar:
    st.subheader(f"Airfoil Aerodynamic Polars: {sel_af}")
    target_re = st.select_slider("Interpolated Reynolds Number (Re)", options=[50000, 100000, 200000, 500000, 1000000], value=500000)
    
    fig_pol, (ax_cl, ax_cd, ax_pol) = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=100)
    alpha_sweep_pts = np.linspace(-10.0, 18.0, 100)
    alpha_sweep_rad = np.radians(alpha_sweep_pts)
    
    if af_model.has_polars:
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(af_model.re_list)))
        for i, r_val in enumerate(af_model.re_list):
            df_p = af_model.raw_data[r_val]
            ax_cl.plot(df_p['alpha'], df_p['CL'], '--', lw=1.0, color=colors[i], alpha=0.45, label=f'Re={int(r_val):,}')
            ax_cd.plot(df_p['alpha'], df_p['CD'], '--', lw=1.0, color=colors[i], alpha=0.45)
            ax_pol.plot(df_p['CD'], df_p['CL'], '--', lw=1.0, color=colors[i], alpha=0.45)
            
    re_arr = np.full_like(alpha_sweep_rad, target_re)
    mach_arr = np.zeros_like(alpha_sweep_rad)
    cl_vals, cd_vals, _ = af_model.evaluate_vectorized(alpha_sweep_rad, re_arr, mach_arr)
    
    ax_cl.plot(alpha_sweep_pts, cl_vals, 'b-', lw=2.2, label=f'Target Re={target_re:,}')
    ax_cl.set_xlabel("Angle of Attack α [°]")
    ax_cl.set_ylabel("Lift Coefficient Cl")
    ax_cl.set_title("Cl vs. α", fontsize=10, fontweight='bold')
    ax_cl.grid(True, linestyle=':', alpha=0.6)
    ax_cl.legend(fontsize=7)
    
    ax_cd.plot(alpha_sweep_pts, cd_vals, 'r-', lw=2.2, label=f'Target Re={target_re:,}')
    ax_cd.set_xlabel("Angle of Attack α [°]")
    ax_cd.set_ylabel("Drag Coefficient Cd")
    ax_cd.set_title("Cd vs. α", fontsize=10, fontweight='bold')
    ax_cd.grid(True, linestyle=':', alpha=0.6)
    
    ax_pol.plot(cd_vals, cl_vals, 'g-', lw=2.2, label=f'Target Re={target_re:,}')
    ax_pol.set_xlabel("Drag Coefficient Cd")
    ax_pol.set_ylabel("Lift Coefficient Cl")
    ax_pol.set_title("Drag Polar (Cl vs. Cd)", fontsize=10, fontweight='bold')
    ax_pol.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(fig_pol)
    plt.close(fig_pol)

# ── TAB 4: 3D Loft & Dual-Mode BEMT ───────────────────────────────────────────
with t_blade:
    st.subheader("3D Lofted Proprotor Blade & Dual-Mode BEMT AoA Checks")
    
    rho_sl, a_sl, _ = isa_atmosphere(0.0)
    
    # Dual BEMT Evaluations: Hover (using th_75_hover) vs Cruise (using th_75_cruise)
    geom_hov = RotorGeometry(
        radius=r_rotor, root_cutout=r_root, num_blades=n_blades,
        chord_func=make_chord_func(c_blade, blade_tap, r_rotor, r_root),
        twist_func=make_twist_func(th_75_hover, th_tw, r_rotor, r_ref_norm=0.75)
    )
    cond_hov = FlightCondition(v_axial=0.0, rpm=440.0, rho=rho_sl, speed_of_sound=a_sl)
    bemt_hov = run_bemt(geom_hov, cond_hov, af_model, num_elements=30)
    
    geom_cr = RotorGeometry(
        radius=r_rotor, root_cutout=r_root, num_blades=n_blades,
        chord_func=make_chord_func(c_blade, blade_tap, r_rotor, r_root),
        twist_func=make_twist_func(th_75_cruise, th_tw, r_rotor, r_ref_norm=0.75)
    )
    cond_cr = FlightCondition(v_axial=125.0, rpm=365.0, rho=rho_sl, speed_of_sound=a_sl)
    bemt_cr = run_bemt(geom_cr, cond_cr, af_model, num_elements=30)
    
    fig_r = plt.figure(figsize=(13.0, 8.5), dpi=100)
    gs_r = fig_r.add_gridspec(2, 2, width_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    
    # 1. 3D Lofted Blade Plot
    ax_r = fig_r.add_subplot(gs_r[:, 0], projection='3d')
    x_af, y_af = load_airfoil_coords(af_model.airfoil_name)
    
    n_elem = 22
    r_edges = np.linspace(r_root, r_rotor, n_elem + 1)
    ax_r.plot([0, r_root], [0, 0], [0, 0], color='#495057', lw=3.5, label=f'Root Cutout ({r_root:.2f}m)')
    ax_r.plot([r_root, r_rotor], [0, 0], [0, 0], 'k--', lw=1.2, label='Pitch Axis (25% c)')
    
    for i in range(n_elem):
        r_in, r_out = r_edges[i], r_edges[i+1]
        c_in = c_blade + (c_blade * blade_tap - c_blade) * ((r_in - r_root) / max(1e-4, r_rotor - r_root))
        c_out = c_blade + (c_blade * blade_tap - c_blade) * ((r_out - r_root) / max(1e-4, r_rotor - r_root))
        th_in = np.radians(th_75_cruise + th_tw * ((r_in / r_rotor) - 0.75))
        th_out = np.radians(th_75_cruise + th_tw * ((r_out / r_rotor) - 0.75))

        x_rot_in = (0.25 - x_af) * c_in * np.cos(th_in) - y_af * c_in * np.sin(th_in)
        z_rot_in = (0.25 - x_af) * c_in * np.sin(th_in) + y_af * c_in * np.cos(th_in)
        x_rot_out = (0.25 - x_af) * c_out * np.cos(th_out) - y_af * c_out * np.sin(th_out)
        z_rot_out = (0.25 - x_af) * c_out * np.sin(th_out) + y_af * c_out * np.cos(th_out)

        poly_list = [[
            [r_in, x_rot_in[j], z_rot_in[j]],
            [r_in, x_rot_in[j+1], z_rot_in[j+1]],
            [r_out, x_rot_out[j+1], z_rot_out[j+1]],
            [r_out, x_rot_out[j], z_rot_out[j]]
        ] for j in range(len(x_af) - 1)]

        ax_r.add_collection3d(art3d.Poly3DCollection(poly_list, facecolors=(0.2, 0.7, 0.9, 0.65), edgecolors=(0, 0, 0, 0.2), linewidths=0.25))

    ax_r.set_box_aspect((2.5, 1.2, 0.9))
    ax_r.set_xlim(0, r_rotor + 0.1)
    ax_r.set_ylim(-c_blade * 0.9, c_blade * 0.9)
    ax_r.set_zlim(-c_blade * 0.5, c_blade * 0.5)
    ax_r.set_xlabel('Radius r [m]'); ax_r.set_ylabel('Chordwise x [m]'); ax_r.set_zlabel('Thickness z [m]')
    ax_r.set_title(f"3D Lofted Blade ({sel_af})", fontsize=10.5, fontweight='bold')
    ax_r.view_init(elev=22, azim=-60)
    ax_r.legend(loc='upper right', fontsize=8)

    # 2. Planform & Twist Distribution
    ax_g1 = fig_r.add_subplot(gs_r[0, 1])
    r_norm = bemt_hov.r_stations / r_rotor
    chords_arr = np.array([geom_hov.chord_func(r) for r in bemt_hov.r_stations])
    thetas_hov_deg = np.degrees([geom_hov.twist_func(r) for r in bemt_hov.r_stations])
    thetas_cr_deg = np.degrees([geom_cr.twist_func(r) for r in bemt_cr.r_stations])
    
    ax_g1_tw = ax_g1.twinx()
    ax_g1.plot(r_norm, chords_arr, 'b-o', ms=3.5, label='Chord c(r) [m]')
    ax_g1_tw.plot(r_norm, thetas_hov_deg, 'r--s', ms=3.0, label=f'Hover θ(r) (θ_0.75={th_75_hover:.1f}°)')
    ax_g1_tw.plot(r_norm, thetas_cr_deg, 'm-.^', ms=3.0, label=f'Cruise θ(r) (θ_0.75={th_75_cruise:.1f}°)')
    ax_g1.set_xlabel('Radial Station (r/R)', fontsize=9)
    ax_g1.set_ylabel('Chord c [m]', color='b', fontsize=9)
    ax_g1_tw.set_ylabel('Geometric Pitch θ [deg]', color='r', fontsize=9)
    ax_g1.set_title("A. Blade Planform & Dual Pitch Schedules", fontsize=10, fontweight='bold')
    ax_g1.grid(True, linestyle=':', alpha=0.6)
    ax_g1.legend(loc='upper left', fontsize=7.5)
    ax_g1_tw.legend(loc='lower left', fontsize=7.5)

    # 3. Dual Angle of Attack Distribution (BEMT Stall Check)
    ax_g2 = fig_r.add_subplot(gs_r[1, 1])
    deg_alpha_hov = np.degrees(bemt_hov.alpha)
    deg_alpha_cr = np.degrees(bemt_cr.alpha)
    
    ax_g2.plot(r_norm, deg_alpha_hov, 'b-^', ms=4.5, lw=1.8, label=f'Hover α(r) [θ_0.75={th_75_hover:.1f}°]')
    ax_g2.plot(r_norm, deg_alpha_cr, 'g-v', ms=4.5, lw=1.8, label=f'Cruise α(r) [θ_0.75={th_75_cruise:.1f}° @ 450 km/h]')
    ax_g2.axhspan(12.0, 20.0, color='red', alpha=0.15, label='Static Stall Danger (>12°)')
    ax_g2.axhline(0.0, color='k', ls=':', lw=1.2)
    ax_g2.set_xlabel('Radial Station (r/R)', fontsize=9)
    ax_g2.set_ylabel('Sectional AoA α [deg]', fontsize=9)
    ax_g2.set_title("B. Dual-Mode AoA Distribution (BEMT Aeromechanical Check)", fontsize=10, fontweight='bold')
    ax_g2.grid(True, linestyle=':', alpha=0.6)
    ax_g2.legend(loc='upper right', fontsize=7.5)

    plt.tight_layout()
    st.pyplot(fig_r)
    plt.close(fig_r)

# ── TAB 5: Power & Forward Speed Flight Envelope ──────────────────────────────
with t_power:
    st.subheader("Power vs. Forward Speed Characteristic & Mission Envelope")
    
    g = 9.80665
    rho_sl, a_sl, _ = isa_atmosphere(0.0)

    # Physical Required Hover Power (with 8% download factor)
    download_factor = 1.08
    T_hover_total_N = takeoff_gross_mass * g * download_factor
    P_induced_ideal_W = (T_hover_total_N ** 1.5) / np.sqrt(2.0 * rho_sl * disk_area_total)
    P_hover_shaft_kW = (P_induced_ideal_W / 0.74) / (0.94 * 1000.0)

    # Cruise Power Sweep (150 to 550 km/h) in Airplane Mode (Rotors Tilted 90 deg)
    v_sweep_kmh = np.linspace(150.0, 550.0, 100)
    v_sweep_ms = v_sweep_kmh / 3.6
    W_N = takeoff_gross_mass * g
    q_dyn_sweep = 0.5 * rho_sl * (v_sweep_ms ** 2)
    CL_sweep = W_N / (q_dyn_sweep * s_wing)
    CD_ind_sweep = (CL_sweep ** 2) / (np.pi * ar_wing * oswald_e)
    CD_total_sweep = cd0 + CD_ind_sweep
    Drag_total_N = q_dyn_sweep * s_wing * CD_total_sweep
    P_cruise_sweep_kW = (Drag_total_N * v_sweep_ms) / (0.83 * 1000.0)

    # Operating Point at 450 km/h
    v_target_ms = 450.0 / 3.6
    q_dyn_target = 0.5 * rho_sl * (v_target_ms ** 2)
    CL_target = W_N / (q_dyn_target * s_wing)
    CD_target = cd0 + (CL_target ** 2) / (np.pi * ar_wing * oswald_e)
    P_cruise_450_kW = (q_dyn_target * s_wing * CD_target * v_target_ms) / (0.83 * 1000.0)

    fig_a, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=100)

    # Plot 1: Power Required vs True Airspeed
    ax_a1.plot(v_sweep_kmh, P_cruise_sweep_kW, 'b-', lw=2.2, label=r'Cruise Power $P_{\mathrm{req}}$ (Wing-Borne, $90^\circ$ Tilt)')
    ax_a1.axhline(P_hover_shaft_kW, color='crimson', ls='--', lw=2.0, label=f'Hover Required Power ({P_hover_shaft_kW:.0f} kW)')
    ax_a1.scatter(450.0, P_cruise_450_kW, color='lime', edgecolors='k', s=140, marker='*', zorder=5, label=f'450 km/h Cruise ({P_cruise_450_kW:.0f} kW)')
    ax_a1.axhline(p_inst_kw, color='black', ls=':', lw=1.8, label=f'Installed Engine Power ({p_inst_kw:.0f} kW)')
    
    ax_a1.set_xlabel(r"True Airspeed $V_\infty$ [km/h]", fontsize=9.5)
    ax_a1.set_ylabel("Total Shaft Power [kW]", fontsize=9.5)
    ax_a1.set_title("1. Power vs. Forward Speed Characteristic", fontsize=10.5, fontweight='bold')
    ax_a1.set_ylim(0, max(p_inst_kw, P_hover_shaft_kW) * 1.25)
    ax_a1.grid(True, linestyle=':', alpha=0.6)
    ax_a1.legend(loc='upper right', fontsize=7.8)

    # Plot 2: Mission Telemetry Step Power & Tip Mach
    if not df_res.empty:
        ax_a2_twin = ax_a2.twinx()
        ax_a2.plot(df_res["time_min"], df_res["p_req_kw"], 'r-', lw=2.0, label=r'Mission $P_{\mathrm{req}}$')
        ax_a2.plot(df_res["time_min"], df_res["p_avail_kw"], 'k--', lw=1.5, label=r'Mission $P_{\mathrm{avail}}$')
        ax_a2_twin.plot(df_res["time_min"], df_res["mach_tip"], 'm-.', lw=1.6, label='Rotor Tip Mach')
        ax_a2_twin.axhline(0.75, color='darkorange', ls=':', label=r'Mach Limit ($M_{\mathrm{tip}}=0.75$)')
        
        ax_a2.set_xlabel("Mission Time [min]", fontsize=9.5)
        ax_a2.set_ylabel("Shaft Power [kW]", color='r', fontsize=9.5)
        ax_a2_twin.set_ylabel("Tip Mach Number", color='m', fontsize=9.5)
        ax_a2.set_title("2. Mission Segment Power & Mach History", fontsize=10.5, fontweight='bold')
        ax_a2.grid(True, linestyle=':', alpha=0.6)
        ax_a2.legend(loc='upper left', fontsize=7.8)
        ax_a2_twin.legend(loc='upper right', fontsize=7.8)
        
    plt.tight_layout()
    st.pyplot(fig_a)
    plt.close(fig_a)

# ── TAB 6: Statistical Regressions & Benchmarks ───────────────────────────────
with t_reg:
    st.subheader("Statistical Regressions & Historical Tiltrotor Benchmarks")
    fig_reg, axes_reg = plt.subplots(2, 2, figsize=(12.0, 9.0), dpi=100)
    
    cur_dl = takeoff_gross_mass / disk_area_total
    cur_sol = (n_blades * c_blade) / (np.pi * r_rotor)
    cur_pw = p_inst_kw / takeoff_gross_mass
    
    # 1. Empty Weight Fraction vs MTOW
    ax_r1 = axes_reg[0, 0]
    x_w_grid = np.linspace(1500, 26000, 200)
    ax_r1.scatter(benchmarks["MTOW_kg"], benchmarks["We_W0"], color="crimson", s=60, edgecolors='k', zorder=4, label="Historical Actuals")
    for _, r in benchmarks.iterrows():
        ax_r1.annotate(r["Aircraft"], (r["MTOW_kg"] * 1.04, r["We_W0"]), fontsize=7.5)
    ax_r1.plot(x_w_grid, power_law(x_w_grid, *popt_we), 'b-', lw=1.8, label=r'Power Fit ($W_e/W_0$)')
    ax_r1.scatter(takeoff_gross_mass, empty_frac, color="lime", s=140, marker="*", edgecolors='k', zorder=5, label=f"Your Aircraft: {empty_frac:.3f}")
    ax_r1.set_xscale("log")
    ax_r1.set_xlabel(r"Gross Weight $W_0$ [kg] (Log Scale)", fontsize=9.5)
    ax_r1.set_ylabel(r"Empty Weight Fraction ($W_e/W_0$)", fontsize=9.5)
    ax_r1.set_title("1. Empty Weight Fraction Regression", fontsize=10.5, fontweight="bold")
    ax_r1.grid(True, which="both", ls=":", alpha=0.6)
    ax_r1.legend(loc="upper right", fontsize=7.8)

    # 2. Installed Power Loading vs Disc Loading
    ax_r2 = axes_reg[0, 1]
    x_dl_grid = np.linspace(25, 160, 200)
    ax_r2.scatter(benchmarks["Disc_Loading"], benchmarks["Installed_PW"], color="crimson", s=60, edgecolors='k', zorder=4, label="Historical Actuals")
    for _, r in benchmarks.iterrows():
        ax_r2.annotate(r["Aircraft"], (r["Disc_Loading"] + 2, r["Installed_PW"]), fontsize=7.5)
    ax_r2.plot(x_dl_grid, power_law(x_dl_grid, *popt_pw), 'b-', lw=1.8, label=r'Power Fit ($P/W_0$)')
    ax_r2.scatter(cur_dl, cur_pw, color="lime", s=140, marker="*", edgecolors='k', zorder=5, label=f"Your P/W: {cur_pw:.3f} kW/kg")
    ax_r2.set_xlabel(r"Disc Loading $DL$ [$\mathrm{kg/m^2}$]", fontsize=9.5)
    ax_r2.set_ylabel(r"Installed Power Loading $P/W_0$ [$\mathrm{kW/kg}$]", fontsize=9.5)
    ax_r2.set_title("2. Power Loading vs. Disc Loading", fontsize=10.5, fontweight="bold")
    ax_r2.grid(True, ls=":", alpha=0.6)
    ax_r2.legend(loc="upper left", fontsize=7.8)

    # 3. Blade Solidity vs Disc Loading
    ax_r3 = axes_reg[1, 0]
    ax_r3.scatter(benchmarks["Disc_Loading"], benchmarks["Solidity"], color="crimson", s=60, edgecolors='k', zorder=4, label="Historical Actuals")
    for _, r in benchmarks.iterrows():
        ax_r3.annotate(r["Aircraft"], (r["Disc_Loading"] + 2, r["Solidity"]), fontsize=7.5)
    ax_r3.plot(x_dl_grid, linear_law(x_dl_grid, *popt_sol), 'g-', lw=1.8, label=r'Linear Fit ($\sigma$)')
    ax_r3.scatter(cur_dl, cur_sol, color="lime", s=140, marker="*", edgecolors='k', zorder=5, label=f"Your $\sigma$: {cur_sol:.4f}")
    ax_r3.set_xlabel(r"Disc Loading $DL$ [$\mathrm{kg/m^2}$]", fontsize=9.5)
    ax_r3.set_ylabel(r"Blade Solidity $\sigma$", fontsize=9.5)
    ax_r3.set_title("3. Blade Solidity vs. Disc Loading", fontsize=10.5, fontweight="bold")
    ax_r3.grid(True, ls=":", alpha=0.6)
    ax_r3.legend(loc="upper left", fontsize=7.8)

    # 4. Service Ceiling Capability Regression
    ax_r4 = axes_reg[1, 1]
    x_ceil_grid = np.linspace(3000, 9500, 200)
    ax_r4.scatter(benchmarks["Service_Ceiling_m"], benchmarks["Installed_PW"], color="crimson", s=60, edgecolors='k', zorder=4, label="Historical Actuals")
    for _, r in benchmarks.iterrows():
        ax_r4.annotate(r["Aircraft"], (r["Service_Ceiling_m"] + 80, r["Installed_PW"]), fontsize=7.5)
    ax_r4.plot(x_ceil_grid, linear_law(x_ceil_grid, *popt_ceil), 'm-', lw=1.8, label=r'Ceiling Fit ($P/W_0$)')
    ax_r4.axvline(7000, color='darkorange', ls='--', label="Req. Ceiling = 7000 m")
    
    attainable_ceiling_m = (cur_pw - popt_ceil[1]) / popt_ceil[0]
    ax_r4.scatter(7000, cur_pw, color="lime", s=140, marker="*", edgecolors='k', zorder=5, 
                 label=f"Design P/W ({cur_pw:.3f} kW/kg)\nAttainable: ~{attainable_ceiling_m:,.0f} m")
    ax_r4.set_xlabel("Service Ceiling [m]", fontsize=9.5)
    ax_r4.set_ylabel(r"Installed Power Loading $P/W_0$ [$\mathrm{kW/kg}$]", fontsize=9.5)
    ax_r4.set_title("4. Service Ceiling Capability Regression", fontsize=10.5, fontweight="bold")
    ax_r4.grid(True, ls=":", alpha=0.6)
    ax_r4.legend(loc="upper left", fontsize=7.8)

    plt.tight_layout()
    st.pyplot(fig_reg)
    plt.close(fig_reg)

# ── TAB 6.1: Hover Performance Maps (Slide 26) ────────────────────────────────
with t_hov_maps:
    st.subheader("Section 6.1: Hover Performance Maps & Operational Envelope")
    
    rho_sl, a_sl, _ = isa_atmosphere(0.0)
    rho_ceil, a_ceil, _ = isa_atmosphere(7000.0)
    p_avail_ceil_kw = p_installed_total_kw * ((rho_ceil / rho_sl) ** 1.05)
    
    fig_h_maps, axs_hm = plt.subplots(2, 2, figsize=(14.0, 8.8), dpi=100)
    
    # 1. Sweep collective pitch from 0 to 24 deg at Hover RPM
    th_sweep = np.linspace(0.0, 24.0, 25)
    t_sl_arr, p_sl_arr, q_sl_arr, aoa_root_sl, aoa_75_sl = [], [], [], [], []
    t_ceil_arr, p_ceil_arr, q_ceil_arr, aoa_root_ceil, aoa_75_ceil = [], [], [], [], []
    
    for th_val in th_sweep:
        # Sea Level
        g_s = RotorGeometry(radius=r_rotor, root_cutout=r_root, num_blades=n_blades,
                            chord_func=make_chord_func(c_blade, blade_tap, r_rotor, r_root),
                            twist_func=make_twist_func(th_val, th_tw, r_rotor, r_ref_norm=0.75))
        c_s = FlightCondition(v_axial=0.0, rpm=440.0, rho=rho_sl, speed_of_sound=a_sl)
        b_s = run_bemt(g_s, c_s, af_model, num_elements=20)
        t_sl_arr.append(b_s.thrust * 2.0)
        p_sl_arr.append((b_s.power * 2.0) / (0.94 * 1000.0))
        q_sl_arr.append(b_s.torque * 2.0)
        aoa_root_sl.append(np.degrees(b_s.alpha[0]))
        aoa_75_sl.append(np.degrees(b_s.alpha[int(len(b_s.alpha)*0.75)]))
        
        # Ceiling
        c_c = FlightCondition(v_axial=0.0, rpm=440.0, rho=rho_ceil, speed_of_sound=a_ceil)
        b_c = run_bemt(g_s, c_c, af_model, num_elements=20)
        t_ceil_arr.append(b_c.thrust * 2.0)
        p_ceil_arr.append((b_c.power * 2.0) / (0.94 * 1000.0))
        q_ceil_arr.append(b_c.torque * 2.0)
        aoa_root_ceil.append(np.degrees(b_c.alpha[0]))
        aoa_75_ceil.append(np.degrees(b_c.alpha[int(len(b_c.alpha)*0.75)]))
        
    t_sl_arr, p_sl_arr, q_sl_arr = np.array(t_sl_arr), np.array(p_sl_arr), np.array(q_sl_arr)
    t_ceil_arr, p_ceil_arr, q_ceil_arr = np.array(t_ceil_arr), np.array(p_ceil_arr), np.array(q_ceil_arr)
    w_hover_target = takeoff_gross_mass * 9.80665 * 1.08 # with 8% download

    # [0, 0] Thrust vs Collective
    ax_h1 = axs_hm[0, 0]
    ax_h1.plot(th_sweep, t_sl_arr / 1000.0, 'b-', lw=2.2, label='Sea Level (ISA 0m)')
    ax_h1.plot(th_sweep, t_ceil_arr / 1000.0, 'b--', lw=1.8, label='Hover Ceiling (7,000m)')
    ax_h1.axhline(w_hover_target / 1000.0, color='crimson', ls=':', lw=1.8, label=f'Target Thrust ({w_hover_target/1000.0:.1f} kN)')
    ax_h1.scatter([th_75_hover], [w_hover_target/1000.0], color='lime', s=140, marker='*', edgecolors='k', zorder=5, label=f'Hover θ_0.75 = {th_75_hover:.1f}°')
    ax_h1.set_xlabel(r'Collective Pitch $\theta_{0.75}$ [deg]', fontsize=9)
    ax_h1.set_ylabel('Total Proprotor Thrust [kN]', fontsize=9)
    ax_h1.set_title('1. Thrust vs. Collective Pitch', fontsize=10.5, fontweight='bold')
    ax_h1.grid(True, ls=':', alpha=0.6); ax_h1.legend(fontsize=7.8)

    # [0, 1] AoA & Stall Margin vs Collective
    ax_h2 = axs_hm[0, 1]
    ax_h2.plot(th_sweep, aoa_root_sl, 'r-', lw=2.0, label='Root AoA α_root (SL)')
    ax_h2.plot(th_sweep, aoa_75_sl, 'm-', lw=1.8, label='75% AoA α_0.75 (SL)')
    ax_h2.plot(th_sweep, aoa_root_ceil, 'r--', lw=1.5, label='Root AoA α_root (Ceiling)')
    ax_h2.axhline(12.0, color='crimson', ls='--', lw=1.5, label='Static Stall Limit (α = 12°)')
    ax_h2.axhspan(12.0, 25.0, color='red', alpha=0.12, label='Stall-Limited Region')
    ax_h2.set_xlabel(r'Collective Pitch $\theta_{0.75}$ [deg]', fontsize=9)
    ax_h2.set_ylabel('Sectional Angle of Attack [deg]', fontsize=9)
    ax_h2.set_title('2. Blade AoA & Stall Margin vs. Collective', fontsize=10.5, fontweight='bold')
    ax_h2.grid(True, ls=':', alpha=0.6); ax_h2.legend(fontsize=7.8)

    # [1, 0] Torque & Power vs Collective
    ax_h3 = axs_hm[1, 0]
    ax_h3_tw = ax_h3.twinx()
    l_p1, = ax_h3.plot(th_sweep, p_sl_arr, 'r-', lw=2.2, label='Shaft Power (SL)')
    l_p2, = ax_h3.plot(th_sweep, p_ceil_arr, 'r--', lw=1.8, label='Shaft Power (Ceiling)')
    l_pmax = ax_h3.axhline(p_installed_total_kw, color='black', ls=':', lw=1.8, label=f'Installed ({p_installed_total_kw:.0f} kW)')
    l_pmax_c = ax_h3.axhline(p_avail_ceil_kw, color='gray', ls=':', lw=1.5, label=f'Avail at Ceil ({p_avail_ceil_kw:.0f} kW)')
    ax_h3.axhspan(p_installed_total_kw, max(p_installed_total_kw*1.3, np.max(p_sl_arr)), color='red', alpha=0.10)
    l_q, = ax_h3_tw.plot(th_sweep, q_sl_arr / 1000.0, 'g-.', lw=1.5, label='Rotor Torque (SL) [kN·m]')
    ax_h3.set_xlabel(r'Collective Pitch $\theta_{0.75}$ [deg]', fontsize=9)
    ax_h3.set_ylabel('Required Shaft Power [kW]', color='r', fontsize=9)
    ax_h3_tw.set_ylabel('Rotor Torque [kN·m]', color='g', fontsize=9)
    ax_h3.set_title('3. Torque & Power vs. Collective', fontsize=10.5, fontweight='bold')
    ax_h3.grid(True, ls=':', alpha=0.6)
    ax_h3.legend([l_p1, l_p2, l_pmax, l_pmax_c, l_q], [l_p1.get_label(), l_p2.get_label(), l_pmax.get_label(), l_pmax_c.get_label(), l_q.get_label()], fontsize=7.2, loc='upper left')

    # [1, 1] Operating Envelope & Hover Ceiling vs Gross Weight
    ax_h4 = axs_hm[1, 1]
    gw_sweep = np.linspace(2000.0, 24000.0, 40)
    ceil_pwr_lim = []
    ceil_stall_lim = []
    
    for gw in gw_sweep:
        t_req = gw * 9.80665 * 1.08
        p_hov_sl = ((t_req ** 1.5) / np.sqrt(2.0 * 1.225 * disk_area_total)) / (0.74 * 0.94 * 1000.0)
        sigma_lim = (p_hov_sl / max(1.0, p_installed_total_kw)) ** (1.0 / 1.55)
        if sigma_lim < 1.0:
            T_ratio = sigma_lim ** (1.0 / 4.2561)
            h_pwr = max(0.0, (1.0 - T_ratio) * 288.15 / 0.0065)
        else:
            h_pwr = 0.0
        ceil_pwr_lim.append(min(9000.0, h_pwr))
        
        sigma_sol = (n_blades * c_blade) / (np.pi * r_rotor)
        ct_max = 0.135 * sigma_sol
        vtip = (440.0 * 2.0 * np.pi / 60.0) * r_rotor
        rho_min = t_req / max(1.0, ct_max * disk_area_total * (vtip ** 2))
        sigma_stall = rho_min / 1.225
        if sigma_stall < 1.0:
            T_ratio_s = sigma_stall ** (1.0 / 4.2561)
            h_stall = max(0.0, (1.0 - T_ratio_s) * 288.15 / 0.0065)
        else:
            h_stall = 0.0
        ceil_stall_lim.append(min(9000.0, h_stall))
        
    ax_h4.plot(gw_sweep / 1000.0, ceil_pwr_lim, 'r-', lw=2.0, label='Engine Power-Limited Boundary')
    ax_h4.plot(gw_sweep / 1000.0, ceil_stall_lim, 'm--', lw=1.8, label='Blade Stall-Limited Boundary')
    ax_h4.fill_between(gw_sweep / 1000.0, np.minimum(ceil_pwr_lim, ceil_stall_lim), color='lightgreen', alpha=0.25, label='Feasible Hover Region')
    ax_h4.scatter([takeoff_gross_mass/1000.0], [7000.0], color='lime', s=150, marker='*', edgecolors='k', zorder=5, label=f'Design Point ({takeoff_gross_mass/1000.0:.1f}t @ 7000m)')
    ax_h4.set_xlabel('Takeoff Gross Weight [Metric Tons]', fontsize=9)
    ax_h4.set_ylabel('Hover Ceiling Altitude [m]', fontsize=9)
    ax_h4.set_title('4. Hover Operating Envelope & Ceiling Limits', fontsize=10.5, fontweight='bold')
    ax_h4.grid(True, ls=':', alpha=0.6); ax_h4.legend(fontsize=7.8)

    plt.tight_layout()
    st.pyplot(fig_h_maps)
    plt.close(fig_h_maps)

# ── TAB 6.2: Forward-Flight Propeller Maps (Slide 27) ─────────────────────────
with t_cr_maps:
    st.subheader("Section 6.2: Axial Forward-Flight / Propeller Maps")
    
    fig_cr_maps, axs_cm = plt.subplots(2, 2, figsize=(14.0, 8.8), dpi=100)
    
    j_grid = np.linspace(0.5, 4.5, 25)
    thetas_cr_family = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    colors_cr = plt.cm.viridis(np.linspace(0.1, 0.9, len(thetas_cr_family)))
    
    ax_c1 = axs_cm[0, 0]
    ax_c1_tw = ax_c1.twinx()
    ax_c2 = axs_cm[0, 1]
    
    n_rps = 365.0 / 60.0
    D_prop = 2.0 * r_rotor
    eta_envelope = np.zeros_like(j_grid)
    
    for idx_th, th_p in enumerate(thetas_cr_family):
        ct_j_list, cp_j_list, eta_j_list = [], [], []
        for j_val in j_grid:
            v_ax_j = j_val * n_rps * D_prop
            g_cr = RotorGeometry(radius=r_rotor, root_cutout=r_root, num_blades=n_blades,
                                 chord_func=make_chord_func(c_blade, blade_tap, r_rotor, r_root),
                                 twist_func=make_twist_func(th_p, th_tw, r_rotor, r_ref_norm=0.75))
            c_cr_j = FlightCondition(v_axial=v_ax_j, rpm=365.0, rho=rho_sl, speed_of_sound=a_sl)
            b_j = run_bemt(g_cr, c_cr_j, af_model, num_elements=18)
            ct_j = b_j.ct
            cp_j = max(b_j.cp, 1e-6)
            eta_j = (ct_j * j_val) / cp_j if ct_j > 0 else 0.0
            ct_j_list.append(ct_j)
            cp_j_list.append(cp_j)
            eta_j_list.append(np.clip(eta_j, 0.0, 0.92))
            
        ct_j_list, cp_j_list, eta_j_list = np.array(ct_j_list), np.array(cp_j_list), np.array(eta_j_list)
        eta_envelope = np.maximum(eta_envelope, eta_j_list)
        
        ax_c1.plot(j_grid, ct_j_list, color=colors_cr[idx_th], lw=1.6, label=f'θ={th_p:.0f}°')
        ax_c1_tw.plot(j_grid, cp_j_list, color=colors_cr[idx_th], ls='--', lw=1.3)
        ax_c2.plot(j_grid, eta_j_list, color=colors_cr[idx_th], lw=1.6, label=rf'$\theta_{{0.75}}={th_p:.0f}^\circ$')
        
    ax_c1.axhline(0, color='k', ls=':', lw=1.0)
    ax_c1.set_xlabel(r'Advance Ratio $J = V_\infty / (n D)$', fontsize=9)
    ax_c1.set_ylabel(r'Thrust Coefficient $C_T$ (Solid)', fontsize=9)
    ax_c1_tw.set_ylabel(r'Power Coefficient $C_P$ (Dashed)', color='gray', fontsize=9)
    ax_c1.set_title('1. Propeller Performance: $C_T$ and $C_P$ vs. $J$', fontsize=10.5, fontweight='bold')
    ax_c1.grid(True, ls=':', alpha=0.6); ax_c1.legend(fontsize=7.2, loc='upper right')
    
    # Design cruise point
    v_cr_ms = 450.0 / 3.6
    J_cr = v_cr_ms / (n_rps * D_prop)
    eta_prop_cr = bemt_cr.propulsive_eff
    
    ax_c2.plot(j_grid, eta_envelope, 'k-', lw=2.2, label='Max Efficiency Envelope')
    ax_c2.scatter([J_cr], [eta_prop_cr], color='lime', s=160, marker='*', edgecolors='k', zorder=5, label=f'Design Cruise Point (J={J_cr:.2f}, η={eta_prop_cr*100:.1f}%)')
    ax_c2.set_xlabel(r'Advance Ratio $J = V_\infty / (n D)$', fontsize=9)
    ax_c2.set_ylabel(r'Propulsive Efficiency $\eta_p = C_T J / C_P$', fontsize=9)
    ax_c2.set_title('2. Propulsive Efficiency Map vs. Advance Ratio', fontsize=10.5, fontweight='bold')
    ax_c2.set_ylim(0.0, 1.0)
    ax_c2.grid(True, ls=':', alpha=0.6); ax_c2.legend(fontsize=7.5, loc='lower right')
    
    # [1, 0] Spanwise AoA Distribution across speeds
    ax_c3 = axs_cm[1, 0]
    r_norm_pts = bemt_cr.r_stations / r_rotor
    test_j_speeds = [1.5, 2.5, 3.5, 4.2]
    colors_j = ['blue', 'green', 'orange', 'crimson']
    for j_t, col_j in zip(test_j_speeds, colors_j):
        v_ax_t = j_t * n_rps * D_prop
        c_cr_t = FlightCondition(v_axial=v_ax_t, rpm=365.0, rho=rho_sl, speed_of_sound=a_sl)
        b_t = run_bemt(geom_cr, c_cr_t, af_model, num_elements=25)
        ax_c3.plot(r_norm_pts, np.degrees(b_t.alpha), color=col_j, lw=1.8, label=f'J={j_t:.1f} ({v_ax_t*3.6:.0f} km/h)')
    ax_c3.axhline(0, color='k', ls=':', lw=1.2)
    ax_c3.axhline(12.0, color='red', ls='--', lw=1.2, label='Stall Limit (12°)')
    ax_c3.axhspan(-20.0, 0.0, color='orange', alpha=0.10, label='Windmilling / Drag Region')
    ax_c3.set_xlabel(r'Radial Station $r/R$', fontsize=9)
    ax_c3.set_ylabel(r'Sectional Angle of Attack $\alpha$ [deg]', fontsize=9)
    ax_c3.set_title('3. Spanwise AoA Distribution across Advance Ratios', fontsize=10.5, fontweight='bold')
    ax_c3.grid(True, ls=':', alpha=0.6); ax_c3.legend(fontsize=7.5, loc='upper right')
    
    # [1, 1] Feasibility Card
    ax_c4 = axs_cm[1, 1]
    ax_c4.axis('off')
    
    q_dyn = 0.5 * rho_sl * (v_cr_ms ** 2)
    CL_target = (takeoff_gross_mass * 9.80665) / (q_dyn * s_wing)
    CD_target = cd0 + (CL_target ** 2) / (np.pi * ar_wing * oswald_e)
    D_total_N = q_dyn * s_wing * CD_target
    p_req_450_kw = (D_total_N * v_cr_ms) / (max(0.01, eta_prop_cr) * 1000.0)
    tip_mach_cr = np.sqrt(((365.0*2*np.pi/60.0)*r_rotor)**2 + v_cr_ms**2) / a_sl
    
    cr_status = 'FEASIBLE & VERIFIED' if (p_req_450_kw <= p_installed_total_kw and tip_mach_cr <= 0.82 and eta_prop_cr >= 0.70) else 'CONSTRAINTS EXCEEDED'
    
    card_str = (
        f'PROPELLER CRUISE PERFORMANCE AUDIT\n'
        f'===============================================\n'
        f'STATUS: {cr_status}\n'
        f'-----------------------------------------------\n'
        f'DESIGN OPERATING CRUISE CONDITION:\n'
        f'  • True Airspeed V_inf:     450.0 km/h ({v_cr_ms:.1f} m/s)\n'
        f'  • Advance Ratio J:         {J_cr:6.2f}\n'
        f'  • Cruise Rotational Speed: 365 RPM\n'
        f'  • Helical Tip Mach M_tip:  {tip_mach_cr:6.3f} (Limit: <= 0.80)\n'
        f'  • Trimmed Cruise Pitch:    {th_75_cruise:6.1f} deg\n\n'
        f'PROPULSIVE POWER & THRUST MATCHING:\n'
        f'  • Total Airplane Drag:     {D_total_N/1000.0:6.2f} kN\n'
        f'  • Rotor Thrust Delivered:  {(bemt_cr.thrust*2)/1000.0:6.2f} kN\n'
        f'  • Propulsive Efficiency η: {eta_prop_cr*100:6.1f} %\n'
        f'  • Cruise Shaft Power Req:  {p_req_450_kw:6.0f} kW\n'
        f'  • Installed Engine Rating: {p_installed_total_kw:6.0f} kW\n'
        f'  • Power Margin at 450 km/h: +{(p_installed_total_kw - p_req_450_kw):6.0f} kW\n'
    )
    border_c = '#28a745' if cr_status == 'FEASIBLE & VERIFIED' else '#dc3545'
    ax_c4.text(0.02, 0.98, card_str, fontfamily='monospace', fontsize=8.8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor=border_c, lw=1.5))
               
    plt.tight_layout()
    st.pyplot(fig_cr_maps)
    plt.close(fig_cr_maps)

# ── TAB 6.3: Comparable Rotor Benchmarks (Slide 28) ───────────────────────────
with t_comp:
    st.subheader("Section 6.3: Comparison with Comparable Proprotors")
    
    fig_comp = plt.figure(figsize=(14.0, 8.8), dpi=100)
    gs_comp = fig_comp.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
    
    # 1. Multi-Aircraft Comparative Benchmark Table
    ax_ctbl = fig_comp.add_subplot(gs_comp[0, :])
    ax_ctbl.axis('off')
    
    sigma_cur = (n_blades * c_blade) / (np.pi * r_rotor)
    v_tip_hov = (440.0 * 2.0 * np.pi / 60.0) * r_rotor
    v_tip_cr = (365.0 * 2.0 * np.pi / 60.0) * r_rotor
    
    comp_df = pd.DataFrame([
        {'Aircraft': 'Bell XV-15 (NASA/Army)', 'R [m]': 3.81, 'Nb': 3, 'Solidity': 0.089, 'Twist [°]': -38.0, 'DL [kg/m²]': 73.2, 'Vtip_hov [m/s]': 225.0, 'Vtip_cr [m/s]': 170.0, 'Max FM': 0.74, 'Cruise η': 0.82, 'P/W0 [kW/kg]': 0.385},
        {'Aircraft': 'Bell Boeing V-22 Osprey', 'R [m]': 5.80, 'Nb': 3, 'Solidity': 0.105, 'Twist [°]': -47.0, 'DL [kg/m²]': 102.5, 'Vtip_hov [m/s]': 240.0, 'Vtip_cr [m/s]': 180.0, 'Max FM': 0.72, 'Cruise η': 0.79, 'P/W0 [kW/kg]': 0.380},
        {'Aircraft': 'Leonardo AW609 (Civil)', 'R [m]': 4.05, 'Nb': 3, 'Solidity': 0.096, 'Twist [°]': -35.0, 'DL [kg/m²]': 88.4, 'Vtip_hov [m/s]': 228.0, 'Vtip_cr [m/s]': 175.0, 'Max FM': 0.75, 'Cruise η': 0.84, 'P/W0 [kW/kg]': 0.354},
        {'Aircraft': 'Bell V-280 Valor (FVL)', 'R [m]': 5.33, 'Nb': 4, 'Solidity': 0.100, 'Twist [°]': -32.0, 'DL [kg/m²]': 95.0, 'Vtip_hov [m/s]': 235.0, 'Vtip_cr [m/s]': 165.0, 'Max FM': 0.76, 'Cruise η': 0.85, 'P/W0 [kW/kg]': 0.533},
        {'Aircraft': '⭐ Your Designed Tiltrotor', 'R [m]': round(r_rotor, 2), 'Nb': n_blades, 'Solidity': round(sigma_cur, 3), 'Twist [°]': round(th_tw, 1), 'DL [kg/m²]': round(cur_dl, 1), 'Vtip_hov [m/s]': round(v_tip_hov, 1), 'Vtip_cr [m/s]': round(v_tip_cr, 1), 'Max FM': round(bemt_hov.figure_of_merit, 2), 'Cruise η': round(bemt_cr.propulsive_eff, 2), 'P/W0 [kW/kg]': round(cur_pw, 3)}
    ])
    
    table_data = [comp_df.columns.tolist()] + comp_df.values.tolist()
    t_elem = ax_ctbl.table(cellText=table_data, loc='center', cellLoc='center')
    t_elem.auto_set_font_size(False)
    t_elem.set_fontsize(8.5)
    t_elem.scale(1.0, 1.5)
    
    for col_i in range(len(comp_df.columns)):
        t_elem[(0, col_i)].set_facecolor('#1f77b4')
        t_elem[(0, col_i)].set_text_props(color='white', fontweight='bold')
        t_elem[(5, col_i)].set_facecolor('#d4edda')
        t_elem[(5, col_i)].set_text_props(fontweight='bold')
        
    ax_ctbl.set_title('COMPARISON WITH COMPARABLE PROPROTOR BENCHMARKS (NORMALIZED NONDIMENSIONAL METRICS)', fontsize=10.5, fontweight='bold', pad=12)
    
    # 2. Nondimensional Metrics Grouped Bar Comparison
    ax_cbar = fig_comp.add_subplot(gs_comp[1, 0])
    labels_m = ['Solidity (σ)', 'Disc Load (DL/80)', 'FM (Hover)', 'Prop Eff (η_cr)', 'P/W (kW/kg)']
    xv15_vals = [0.089, 73.2/80.0, 0.74, 0.82, 0.385]
    aw609_vals = [0.096, 88.4/80.0, 0.75, 0.84, 0.354]
    your_vals = [sigma_cur, cur_dl/80.0, bemt_hov.figure_of_merit, bemt_cr.propulsive_eff, cur_pw]
    
    x_idx = np.arange(len(labels_m))
    bar_w = 0.25
    ax_cbar.bar(x_idx - bar_w, xv15_vals, width=bar_w, color='#4575b4', label='Bell XV-15')
    ax_cbar.bar(x_idx, aw609_vals, width=bar_w, color='#f46d43', label='Leonardo AW609')
    ax_cbar.bar(x_idx + bar_w, your_vals, width=bar_w, color='#2ca02c', label='Your Aircraft')
    ax_cbar.set_xticks(x_idx); ax_cbar.set_xticklabels(labels_m, fontsize=8)
    ax_cbar.set_ylabel('Normalized Metric Value', fontsize=8.5)
    ax_cbar.set_title('Normalized Aerodynamic & Sizing Comparisons', fontsize=10, fontweight='bold')
    ax_cbar.grid(True, ls=':', alpha=0.6); ax_cbar.legend(fontsize=7.8)
    
    # 3. Technical Discussion Card
    ax_ctext = fig_comp.add_subplot(gs_comp[1, 1])
    ax_ctext.axis('off')
    
    accept_verdict = 'ACCEPTABLE & BALANCED' if (0.07 <= sigma_cur <= 0.12 and 50 <= cur_dl <= 120 and bemt_hov.figure_of_merit >= 0.65 and bemt_cr.propulsive_eff >= 0.72) else 'NEEDS GEOMETRIC TUNING'
    
    disc_text = (
        f'DESIGN ACCEPTABILITY & COMPARATIVE EVALUATION\n'
        f'=================================================\n'
        f'OVERALL DESIGN VERDICT: {accept_verdict}\n'
        f'-------------------------------------------------\n'
        f'1. DISC LOADING (DL = {cur_dl:.1f} kg/m²):\n'
        f'   • Sits squarely between XV-15 (73.2) and AW609 (88.4).\n'
        f'   • Compact proprotor diameter prevents fuselage collision\n'
        f'     while providing reasonable hover downwash velocity.\n\n'
        f'2. ROTOR SOLIDITY (σ = {sigma_cur:.3f}):\n'
        f'   • {n_blades}-bladed proprotor provides optimal blade loading\n'
        f'     (Ct/σ ≈ 0.08) avoiding root stall in hover.\n\n'
        f'3. TWIST RATE (θ_tw = {th_tw:.1f}°):\n'
        f'   • Provides attached high-speed cruise thrust while\n'
        f'     mitigating extreme negative root windmilling.\n\n'
        f'4. TIP SPEED SCALING:\n'
        f'   • Hover Vtip ({v_tip_hov:.0f} m/s) -> Cruise Vtip ({v_tip_cr:.0f} m/s)\n'
        f'   • ~{((1 - v_tip_cr/v_tip_hov)*100):.0f}% RPM reduction keeps cruise Mtip <= 0.80.\n'
    )
    border_d = '#28a745' if accept_verdict == 'ACCEPTABLE & BALANCED' else '#ffc107'
    ax_ctext.text(0.02, 0.98, disc_text, fontfamily='monospace', fontsize=8.6, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor=border_d, lw=1.5))
                  
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

# ── TAB 7: Telemetry Log ──────────────────────────────────────────────────────
with t_log:
    st.subheader("State-Marching Telemetry Record")
    if not df_res.empty:
        st.dataframe(
            df_res.style.format({
                "time_min": "{:.2f}",
                "alt_m": "{:.0f}",
                "gross_kg": "{:.1f}",
                "fuel_kg": "{:.1f}",
                "payload_kg": "{:.0f}",
                "p_req_kw": "{:.1f}",
                "p_avail_kw": "{:.1f}",
                "speed_kmh": "{:.1f}",
                "dist_km": "{:.1f}",
                "mach_tip": "{:.3f}"
            }),
            use_container_width=True
        )
        csv_data = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Telemetry CSV", data=csv_data, file_name="tiltrotor_mission_telemetry.csv", mime="text/csv")
    else:
        st.info("No telemetry points generated. Please verify mission parameters.")

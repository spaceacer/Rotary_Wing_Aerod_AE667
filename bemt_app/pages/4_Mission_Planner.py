import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from enum import Enum
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.airfoil_catalog import get_all_available_airfoils
from core.airfoil_model import AirfoilModel
from core.airfoil_blend import BlendedAirfoil
from core.bemt_solver import run_bemt
from core.geometry_helpers import make_chord_func, make_twist_func
from core.models import FlightCondition, RotorGeometry

st.set_page_config(page_title="Mission Planner", page_icon="🗺️", layout="wide")
st.title("🗺️  Mission Planner & Aircraft Sizing")

# ── Helpers ───────────────────────────────────────────────────────────────────

def isa_atmosphere(altitude_m: float) -> tuple:
    alt = np.clip(altitude_m, 0.0, 11000.0)
    T = 288.15 - 0.0065 * alt
    P = 101325.0 * ((T / 288.15) ** 5.2561)
    rho = P / (287.058 * T)
    a = np.sqrt(1.4 * 287.058 * T)
    return rho, a, T

class SegmentType(Enum):
    HOVER = "Hover"
    VERTICAL_CLIMB = "Vertical Climb"
    VERTICAL_DESCENT = "Vertical Descent"
    CRUISE = "Cruise (Axial)"
    LOITER = "Loiter"

PRESETS = {
    "Standard Takeoff -> Cruise -> Land": [
        {"Name": "Takeoff Hover", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Vertical Climb", "Type": "Vertical Climb", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 2500.0, "Climb [m/s]": 5.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Outbound Cruise", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 200.0, "Speed [km/h]": 420.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "Descent", "Type": "Vertical Descent", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": -3.5, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Landing Hover", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
    ],
    "On-Station Loiter": [
        {"Name": "Takeoff Hover", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Climb", "Type": "Vertical Climb", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 2500.0, "Climb [m/s]": 5.0, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Dash", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 150.0, "Speed [km/h]": 440.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 15.0},
        {"Name": "Loiter", "Type": "Loiter", "Dur [min]": 30.0, "Dist [km]": 0.0, "Speed [km/h]": 260.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": 0.0},
        {"Name": "Return", "Type": "Cruise (Axial)", "Dur [min]": 0.0, "Dist [km]": 150.0, "Speed [km/h]": 420.0, "Alt [m]": 2500.0, "Climb [m/s]": 0.0, "RPM": 365.0, "Wind [km/h]": -15.0},
        {"Name": "Descent", "Type": "Vertical Descent", "Dur [min]": 0.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": -3.5, "RPM": 440.0, "Wind [km/h]": 0.0},
        {"Name": "Landing", "Type": "Hover", "Dur [min]": 1.0, "Dist [km]": 0.0, "Speed [km/h]": 0.0, "Alt [m]": 0.0, "Climb [m/s]": 0.0, "RPM": 440.0, "Wind [km/h]": 0.0},
    ]
}

# ── Sidebar Configurations ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Aircraft Specs")
    
    with st.expander("Airframe & Weights", expanded=True):
        mtow = st.number_input("MTOW [kg]", 2000.0, 25000.0, 10500.0, 100.0)
        empty_frac = st.slider("Empty Mass %", 0.35, 0.75, 0.55, 0.01)
        fuel_frac = st.slider("Fuel Mass %", 0.05, 0.45, 0.25, 0.01)
        payload_mass = st.number_input("Payload [kg]", 0.0, 6000.0, 1440.0, 10.0)
        reserve_fuel = st.number_input("Reserve Fuel [kg]", 50.0, 800.0, 300.0, 10.0)
    
    with st.expander("Twin Rotors (BEMT)", expanded=True):
        num_rotors = 2
        r_rotor = st.slider("Rotor Radius R [m]", 0.5, 8.0, 4.25, 0.1)
        r_root = st.slider("Root Cutout [m]", 0.05, 1.5, 0.45, 0.05)
        n_blades = st.number_input("Blades per Rotor", 2, 8, 3)
        c_blade = st.slider("Blade Chord [m]", 0.05, 1.0, 0.38, 0.01)
        blade_tap = st.slider("Blade Taper", 0.2, 1.5, 0.85, 0.05)
        th_75 = st.slider("Blade Pitch θ_0.75 [°]", 0.0, 30.0, 10.0, 0.5)
        th_tw = st.slider("Blade Twist [°]", -30.0, 5.0, -12.0, 0.5)

    with st.expander("Wing & Fuselage"):
        s_wing = st.slider("Wing Area [m²]", 6.0, 80.0, 32.0, 0.5)
        ar_wing = st.slider("Aspect Ratio", 4.0, 16.0, 7.8, 0.1)
        taper_w = st.slider("Wing Taper", 0.2, 1.0, 0.65, 0.05)
        sweep_deg = st.slider("Wing Sweep [°]", -10.0, 35.0, 2.5, 0.5)
        
        l_fuse = st.slider("Fuse Length [m]", 4.0, 30.0, 14.5, 0.5)
        w_fuse = st.slider("Fuse Width [m]", 0.6, 5.0, 2.1, 0.1)
        vh_tail = st.slider("H-Tail Vol Coeff", 0.2, 1.5, 0.85, 0.05)
        
        cd0 = st.number_input("Airframe CD0", 0.010, 0.060, 0.024, 0.001, format="%.3f")
        oswald_e = st.slider("Oswald Eff (e)", 0.60, 0.95, 0.82, 0.01)
        
    with st.expander("Engine"):
        p_inst_kw = st.number_input("Inst Power [kW]", 500.0, 15000.0, 4800.0, 100.0)
        sfc_hr = st.number_input("SFC [kg/kW/hr]", 0.15, 0.60, 0.285, 0.005, format="%.3f")

# ── Mission Schedule Builder ──────────────────────────────────────────────────
st.subheader("📋 Mission Schedule")
preset_name = st.selectbox("Load Preset:", list(PRESETS.keys()))

if "mission_df" not in st.session_state or st.session_state.get("last_preset") != preset_name:
    st.session_state["mission_df"] = pd.DataFrame(PRESETS[preset_name])
    st.session_state["last_preset"] = preset_name

edited_mission = st.data_editor(
    st.session_state["mission_df"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn(
            "Segment Type", options=[e.value for e in SegmentType], required=True
        )
    }
)
st.session_state["mission_df"] = edited_mission

# Retrieve Airfoil Model from BEMT dashboard state if available, else default to NACA 0012
af_table = st.session_state.get("airfoil_table")
if af_table is not None and not af_table.empty:
    af_names = af_table["Airfoil"].tolist()
    r_stations = af_table["r/R"].tolist()
else:
    af_names = ["NACA 0012", "NACA 0012"]
    r_stations = [0.2, 1.0]

@st.cache_data(show_spinner=False)
def _get_blend(names, r_stats):
    models = [AirfoilModel(name, ncrit_pref=9) for name in names]
    return BlendedAirfoil(r_stats, models)

af_blend = _get_blend(tuple(af_names), tuple(r_stations))

# ── Physics Engine ────────────────────────────────────────────────────────────
if st.button("🚀 Run Mission Simulation", type="primary"):
    with st.spinner("Marching through mission timeline..."):
        m_empty = mtow * empty_frac
        m_fuel_init = mtow * fuel_frac
        takeoff_gross_mass = m_empty + m_fuel_init + payload_mass
        
        b_wing = np.sqrt(s_wing * ar_wing)
        c_root_w = (2.0 * s_wing) / (b_wing * (1.0 + taper_w))
        c_tip_w = c_root_w * taper_w
        mac = (2.0 / 3.0) * c_root_w * (1.0 + taper_w + taper_w**2) / (1.0 + taper_w)
        l_arm = 0.45 * l_fuse
        s_htail = (vh_tail * s_wing * mac) / l_arm
        
        # Collision Check
        y_tip_rotor = 0.5 * b_wing
        fuse_clearance = (y_tip_rotor - r_rotor) - (0.5 * w_fuse)
        collision_warn = fuse_clearance < 0.25
        collision_msg = f"CRITICAL: Rotor blade penetrates fuselage by {abs(fuse_clearance):.2f}m!" if fuse_clearance <= 0 else f"WARNING: Clearance is tight ({fuse_clearance:.2f}m < 0.25m)"
        
        curr_t, curr_alt, curr_fuel, curr_dist = 0.0, 0.0, m_fuel_init, 0.0
        telemetry = []
        failed, fail_msg = False, ""
        dt = 5.0 # seconds per step
        
        for _, leg in edited_mission.iterrows():
            start_alt = curr_alt
            seg_type = leg["Type"]
            dur_m = float(leg["Dur [min]"])
            dist_km = float(leg["Dist [km]"])
            speed_k = float(leg["Speed [km/h]"])
            alt_t = float(leg["Alt [m]"])
            climb_v = float(leg["Climb [m/s]"])
            rpm_val = float(leg["RPM"])
            wind_k = float(leg["Wind [km/h]"])
            
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
            elif seg_type == SegmentType.CRUISE.value:
                v_tas = max(speed_k / 3.6, 15.0)
                v_ground = max(v_tas - (wind_k / 3.6), 5.0)
                dur_s = max(dist_km * 1000.0, 100.0) / v_ground
                alt_target = alt_t
                
            n_steps = max(int(np.ceil(dur_s / dt)), 1)
            step_dt = dur_s / n_steps
            alt_rate = (alt_target - start_alt) / dur_s if dur_s > 0 else 0.0
            
            # Setup BEMT static geometry once
            geom = RotorGeometry(
                radius=r_rotor, root_cutout=r_root, num_blades=n_blades,
                chord_func=make_chord_func(c_blade, blade_tap, r_rotor, r_root),
                twist_func=make_twist_func(th_75, th_tw, r_rotor)
            )
            
            for _ in range(n_steps):
                gross_m = m_empty + curr_fuel + payload_mass
                w_newtons = gross_m * 9.80665
                rho, a_sound, _ = isa_atmosphere(curr_alt)
                
                omega = (rpm_val * 2.0 * np.pi) / 60.0
                v_tip = omega * r_rotor
                m_tip = np.sqrt(v_tip**2 + v_tas**2) / a_sound
                
                if seg_type in [SegmentType.HOVER.value, SegmentType.VERTICAL_CLIMB.value, SegmentType.VERTICAL_DESCENT.value]:
                    v_axial = climb_v if seg_type != SegmentType.HOVER.value else 0.0
                    cond = FlightCondition(v_axial=v_axial, rpm=rpm_val, rho=rho, speed_of_sound=a_sound)
                    bemt_res = run_bemt(geom, cond, af_blend, num_elements=15)
                    p_req_kw = (num_rotors * bemt_res.power) / (0.94 * 1000.0)
                else:
                    q_dyn = 0.5 * rho * (v_tas ** 2)
                    cl_wing = w_newtons / max(q_dyn * s_wing, 1e-4)
                    cd_ind = (cl_wing ** 2) / (np.pi * ar_wing * oswald_e)
                    drag_total = q_dyn * s_wing * (cd0 + cd_ind)
                    p_req_kw = (drag_total * v_tas) / (0.84 * 1000.0)
                    
                p_avail_kw = p_inst_kw * ((rho / 1.225) ** 1.05)
                fuel_burn = p_req_kw * (sfc_hr / 3600.0) * step_dt
                
                if p_req_kw > p_avail_kw:
                    failed, fail_msg = True, f"Power limit ({p_req_kw:.0f}kW > {p_avail_kw:.0f}kW) in {leg['Name']}"
                    break
                if curr_fuel <= reserve_fuel:
                    failed, fail_msg = True, f"Reserve breached ({curr_fuel:.1f}kg <= {reserve_fuel:.1f}kg) in {leg['Name']}"
                    break
                    
                curr_fuel -= fuel_burn
                curr_alt += alt_rate * step_dt
                curr_dist += (v_ground * step_dt) / 1000.0
                curr_t += step_dt
                
                telemetry.append({
                    "time_min": curr_t / 60.0, "leg": leg['Name'], "type": seg_type, "alt_m": curr_alt,
                    "gross_kg": gross_m, "fuel_kg": curr_fuel, "payload_kg": payload_mass,
                    "p_req_kw": p_req_kw, "p_avail_kw": p_avail_kw, "speed_kmh": v_tas * 3.6,
                    "dist_km": curr_dist, "mach_tip": m_tip
                })
                
            if failed:
                break
                
        df_res = pd.DataFrame(telemetry)
        
    # ── Display Results ───────────────────────────────────────────────────────
    if failed:
        st.error(f"**MISSION FAILED:** {fail_msg}")
    else:
        st.success("Mission executed successfully! Payload and reserves intact.")
        
    t1, t2, t3, t4 = st.tabs(["📐 General Arrangement", "📈 Mission Profile", "⚡ Power Envelope", "📊 Telemetry Log"])
    
    with t1:
        st.subheader("Aircraft Geometry & Clearance Check")
        fig_c, (ax_c, ax_card) = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={'width_ratios': [1.3, 1.0]})
        
        # Fuselage
        fx = np.array([0, 0.15*l_fuse, 0.75*l_fuse, l_fuse, 0.75*l_fuse, 0.15*l_fuse, 0])
        fy = np.array([0, 0.5*w_fuse, 0.5*w_fuse, 0, -0.5*w_fuse, -0.5*w_fuse, 0])
        ax_c.fill(fx, fy, color='#e63946' if collision_warn else '#ced4da', alpha=0.85, edgecolor='k', lw=1.5)

        # Wing
        wx_le = 0.35 * l_fuse
        tip_x_off = 0.5 * b_wing * np.tan(np.radians(sweep_deg))
        wx = [wx_le, wx_le + tip_x_off, wx_le + tip_x_off + c_tip_w, wx_le + c_root_w, wx_le + tip_x_off + c_tip_w, wx_le + tip_x_off, wx_le]
        wy = [0, 0.5*b_wing, 0.5*b_wing, 0, -0.5*b_wing, -0.5*b_wing, 0]
        ax_c.fill(wx, wy, color='#9ec5fe', alpha=0.8, edgecolor='blue', lw=1.5)

        # H-Tail
        b_ht = np.sqrt(s_htail * 4.2)
        c_ht = s_htail / b_ht
        hx_le = wx_le + l_arm
        ax_c.fill([hx_le, hx_le, hx_le + c_ht, hx_le + c_ht], [-0.5*b_ht, 0.5*b_ht, 0.5*b_ht, -0.5*b_ht], color='#6c757d', alpha=0.85, edgecolor='k')

        # Rotors
        rcs = [(wx_le + tip_x_off, 0.5*b_wing), (wx_le + tip_x_off, -0.5*b_wing)]
        for rx, ry in rcs:
            col = 'crimson' if collision_warn else 'darkgreen'
            ax_c.add_patch(plt.Circle((rx, ry), r_rotor, color=col, fill=True, alpha=0.15, linestyle='--', lw=1.6))
            ax_c.add_patch(plt.Circle((rx, ry), r_rotor, color=col, fill=False, linestyle='--', lw=1.6))
            ax_c.plot(rx, ry, 'o', color=col, ms=5)

        ax_c.set_aspect('equal')
        ax_c.set_xlim(-1, l_fuse + 2)
        ax_c.set_ylim(-0.55*b_wing - r_rotor, 0.55*b_wing + r_rotor)
        ax_c.set_title(f"Twin Proprotor Layout" + (f"\n[!] CLEARANCE: {fuse_clearance:.2f}m" if collision_warn else ""), 
                       fontsize=10, fontweight='bold', color='crimson' if collision_warn else 'black')
        ax_c.set_xlabel("X [m]"); ax_c.set_ylabel("Y [m]"); ax_c.grid(True, linestyle=':', alpha=0.6)

        # Text Card
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
            f"ROTOR METRICS:\n"
            f"  • Blended Airfoils:  {' → '.join(af_names)}\n"
            f"  • Rotor Radius R:    {r_rotor:6.2f} m\n"
            f"  • Fuselage Clearance:{fuse_clearance:6.2f} m\n"
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

    with t2:
        if not df_res.empty:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["alt_m"], name="Altitude [m]", line=dict(color="blue", width=3)))
            fig_p.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["speed_kmh"], name="Speed [km/h]", yaxis="y2", line=dict(color="orange", width=2, dash="dash")))
            fig_p.update_layout(
                title="Mission Trajectory",
                xaxis_title="Time [minutes]",
                yaxis_title="Altitude [m]",
                yaxis2=dict(title="True Airspeed [km/h]", overlaying="y", side="right"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_p, use_container_width=True)

            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["gross_kg"], name="Gross Mass [kg]", line=dict(color="green", width=3)))
            fig_m.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["fuel_kg"], name="Fuel Remaining [kg]", line=dict(color="red", width=2, dash="dash")))
            fig_m.add_hline(y=reserve_fuel, line_dash="dot", annotation_text="Reserve Limit", annotation_position="bottom right", line_color="red")
            fig_m.update_layout(title="Mass & Fuel Tracking", xaxis_title="Time [minutes]", yaxis_title="Mass [kg]", hovermode="x unified")
            st.plotly_chart(fig_m, use_container_width=True)

    with t3:
        if not df_res.empty:
            fig_pow = go.Figure()
            fig_pow.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["p_req_kw"], name="Power Required", line=dict(color="red", width=3)))
            fig_pow.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["p_avail_kw"], name="Power Available", line=dict(color="black", width=2, dash="dash")))
            fig_pow.update_layout(title="Shaft Power Profile", xaxis_title="Time [minutes]", yaxis_title="Power [kW]", hovermode="x unified")
            st.plotly_chart(fig_pow, use_container_width=True)

            fig_mach = go.Figure()
            fig_mach.add_trace(go.Scatter(x=df_res["time_min"], y=df_res["mach_tip"], name="Rotor Tip Mach", line=dict(color="magenta", width=3)))
            fig_mach.add_hline(y=0.85, line_dash="dot", annotation_text="Mach Limit (0.85)", annotation_position="bottom right", line_color="red")
            fig_mach.update_layout(title="Compressibility Tracking", xaxis_title="Time [minutes]", yaxis_title="Tip Mach", hovermode="x unified")
            st.plotly_chart(fig_mach, use_container_width=True)

    with t4:
        if not df_res.empty:
            st.dataframe(df_res, use_container_width=True)

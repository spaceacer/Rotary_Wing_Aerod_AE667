import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bemt.atmosphere import get_atmosphere
from bemt.rotor import Rotor
from bemt.solver import solve_bemt
from bemt.airfoil import Airfoil, parse_polar_csv
from bemt.geometry_viz import cross_section_figure, engineering_drawing_figure, rotor_3d_figure

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BEMT Rotor Solver",
    page_icon="AE667",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT_TEMPLATE = "plotly_white"
ACCENT = "#2563EB"
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]

st.markdown(
    """
    <style>
    /* Sticky run button at the bottom of the sidebar */
    section[data-testid="stSidebar"] div.stButton {
        position: sticky;
        bottom: 0px;
        padding-bottom: 1rem;
        padding-top: 0.75rem;
        z-index: 999;
        background-color: var(--background-color);
    }
    section[data-testid="stSidebar"] div.stButton button {
        width: 100%;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        opacity: 0.8;
    }
    h1 {
        padding-bottom: 0rem;
    }
    .subtitle {
        opacity: 0.7;
        margin-top: -0.6rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Blade Element Momentum Theory Solver")
st.markdown(
    '<p class="subtitle">Rotary Wing Aerodynamics Course Project — Milestone 1</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("Rotor Geometry", expanded=True):
        n_blades = st.number_input("Number of Blades", min_value=2, max_value=8, value=2, step=1)
        radius = st.number_input("Rotor Radius (m)", min_value=0.1, value=0.762, step=0.1)
        root_cutout = st.number_input(
            "Root Cut-out (m)", min_value=0.0, max_value=radius * 0.9, value=0.125, step=0.01
        )
        chord_root = st.number_input("Root Chord (m)", min_value=0.01, value=0.0508, step=0.01)
        chord_tip = st.number_input("Tip Chord (m)", min_value=0.01, value=0.0508, step=0.01)
        theta_root_deg = st.number_input("Root Twist (deg)", value=0.0, step=1.0)
        theta_tip_deg = st.number_input("Tip Twist (deg)", value=0.0, step=1.0)

    with st.expander("Airfoil Aerodynamics", expanded=True):
        airfoil_mode = st.radio(
            "Airfoil Data Source",
            options=["Default (Knight & Hefner)", "Custom Parameters", "Upload Polar (CSV)"],
            index=0,
            help="Choose the Cl/Cd vs angle-of-attack model used by the solver.",
        )

        airfoil_error = None
        polar_table = None
        custom_params = {}

        if airfoil_mode == "Custom Parameters":
            st.caption("Linear/quadratic model: Cl = Cl_α·(α − α₀), Cd = Cd₀ + k·(α − α₀)²")
            cl_alpha = st.number_input("Lift Curve Slope, Cl_α (per rad)", value=5.75, step=0.05)
            alpha0_deg = st.number_input("Zero-Lift Angle, α₀ (deg)", value=0.0, step=0.5)
            use_cl_max = st.checkbox("Cap maximum Cl (stall limit)", value=False)
            cl_max = st.number_input("Cl_max", value=1.4, step=0.05) if use_cl_max else None
            cd0 = st.number_input("Zero-Lift Drag, Cd₀", value=0.0113, step=0.001, format="%.4f")
            k_cd = st.number_input("Induced Drag Factor, k", value=1.25, step=0.05)
            custom_params = dict(
                cl_alpha=cl_alpha, alpha0=np.radians(alpha0_deg),
                cl_max=cl_max, cd0=cd0, k_cd=k_cd,
            )

        elif airfoil_mode == "Upload Polar (CSV)":
            st.caption("CSV columns required: `alpha` (deg), `cl`, `cd`")
            uploaded_file = st.file_uploader("Upload airfoil polar CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    polar_table = parse_polar_csv(uploaded_file)
                    st.success(f"Loaded {len(polar_table['alpha'])} data points.")
                except ValueError as e:
                    airfoil_error = str(e)
                    st.error(airfoil_error)
            else:
                st.info("Upload a CSV to use this mode. Falling back to default model until then.")

    with st.expander("Operating Conditions", expanded=True):
        collective_deg = st.number_input("Collective Pitch at 75%R (deg)", value=8.0, step=1.0)
        rpm = st.number_input("Rotational Speed (RPM)", min_value=100.0, value=2000.0, step=100.0)
        v_climb = st.number_input("Axial Velocity (m/s) (Climb/Forward)", value=0.0, step=1.0)

    with st.expander("Atmosphere", expanded=False):
        altitude_m = st.number_input("Altitude (m)", value=0.0, step=100.0)
        delta_T_ISA = st.number_input("ISA Temp Offset (K)", value=0.0, step=1.0)

    with st.expander("Solver Settings", expanded=False):
        n_elements = st.number_input("Number of Blade Elements", min_value=10, value=50, step=10)
        tip_loss = st.checkbox("Include Prandtl Tip Loss", value=True)

    run = st.button("▶  Run Solver", type="primary")

# Process inputs
theta_root = np.radians(theta_root_deg)
theta_tip = np.radians(theta_tip_deg)
collective_rad = np.radians(collective_deg)

# Build the airfoil model from sidebar selection
if airfoil_mode == "Custom Parameters":
    airfoil = Airfoil(mode="custom", params=custom_params)
elif airfoil_mode == "Upload Polar (CSV)" and polar_table is not None:
    airfoil = Airfoil(mode="uploaded", polar_table=polar_table)
else:
    airfoil = Airfoil(mode="default")

rotor = Rotor(n_blades, radius, root_cutout, chord_root, chord_tip, theta_root, theta_tip, airfoil=airfoil)

# ---------------------------------------------------------------------------
# Helper plots
# ---------------------------------------------------------------------------
def make_planform_figure(rotor: Rotor):
    r = np.linspace(rotor.root_cutout, rotor.radius, 100)
    chord = np.array([rotor.get_chord(ri) for ri in r])
    twist_deg = np.degrees([rotor.get_twist(ri) for ri in r])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=r, y=chord, name="Chord (m)", line=dict(color=COLORS[0], width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=r, y=twist_deg, name="Twist (deg)",
            line=dict(color=COLORS[1], width=3, dash="dash"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Blade Planform: Chord & Twist Distribution",
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Radial Station r (m)")
    fig.update_yaxes(title_text="Chord (m)", secondary_y=False)
    fig.update_yaxes(title_text="Twist (deg)", secondary_y=True)
    return fig


def make_polar_figure(rotor: Rotor):
    # For uploaded polars, plot over the actual data range; otherwise a standard sweep.
    if rotor.airfoil.mode == "uploaded" and rotor.airfoil.polar_table is not None:
        alpha_rad = np.linspace(
            rotor.airfoil.polar_table["alpha"].min(),
            rotor.airfoil.polar_table["alpha"].max(),
            200,
        )
        alpha_deg = np.degrees(alpha_rad)
    else:
        alpha_deg = np.linspace(-10, 15, 200)
        alpha_rad = np.radians(alpha_deg)
    cl, cd = rotor.get_cl_cd(alpha_rad)

    mode_label = {"default": "Default (Knight & Hefner)", "custom": "Custom Parameters", "uploaded": "Uploaded Polar"}
    subtitle = mode_label.get(rotor.airfoil.mode, rotor.airfoil.mode)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=alpha_deg, y=cl, name="Cl", line=dict(color=COLORS[2], width=3)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=alpha_deg, y=cd, name="Cd", line=dict(color=COLORS[3], width=3)),
        secondary_y=True,
    )
    if rotor.airfoil.mode == "uploaded" and rotor.airfoil.polar_table is not None:
        fig.add_trace(
            go.Scatter(
                x=np.degrees(rotor.airfoil.polar_table["alpha"]), y=rotor.airfoil.polar_table["cl"],
                mode="markers", name="Cl (data)", marker=dict(color=COLORS[2], size=7, symbol="circle-open"),
            ),
            secondary_y=False,
        )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=f"Airfoil Model: Cl / Cd vs Angle of Attack — {subtitle}",
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Angle of Attack (deg)")
    fig.update_yaxes(title_text="Cl", secondary_y=False)
    fig.update_yaxes(title_text="Cd", secondary_y=True)
    return fig


def make_radial_distributions_figure(df: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Sectional Thrust", "Sectional Torque",
            "Inflow Ratio (λ)", "Angle of Attack",
        ),
    )

    fig.add_trace(
        go.Scatter(x=df["r/R"], y=df["dT (N)"], line=dict(color=COLORS[0], width=2.5), name="dT"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["r/R"], y=df["dQ (N-m)"], line=dict(color=COLORS[1], width=2.5), name="dQ"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["r/R"], y=df["Inflow Ratio (lambda)"], line=dict(color=COLORS[2], width=2.5), name="λ"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["r/R"], y=df["Angle of Attack (deg)"], line=dict(color=COLORS[4], width=2.5), name="α"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="r/R", row=1, col=1)
    fig.update_xaxes(title_text="r/R", row=1, col=2)
    fig.update_xaxes(title_text="r/R", row=2, col=1)
    fig.update_xaxes(title_text="r/R", row=2, col=2)
    fig.update_yaxes(title_text="dT (N)", row=1, col=1)
    fig.update_yaxes(title_text="dQ (N·m)", row=1, col=2)
    fig.update_yaxes(title_text="λ", row=2, col=1)
    fig.update_yaxes(title_text="α (deg)", row=2, col=2)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=650,
        showlegend=False,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


def make_combined_loading_figure(df: pd.DataFrame):
    """dT overlaid with inflow angle on a secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["r/R"], y=df["dT (N)"], name="dT (N)",
            fill="tozeroy", line=dict(color=COLORS[0], width=2.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["r/R"], y=df["Inflow Angle (deg)"], name="Inflow Angle φ (deg)",
            line=dict(color=COLORS[3], width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Sectional Thrust Loading vs Inflow Angle",
        height=380,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="r/R")
    fig.update_yaxes(title_text="dT (N)", secondary_y=False)
    fig.update_yaxes(title_text="φ (deg)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_home, tab_overview, tab_results = st.tabs(["🏠 Home", "📐 Rotor Overview", "📊 Solver Results"])

with tab_home:
    st.caption(
        "Geometric visualization of the current rotor configuration — cross-section, "
        "annotated engineering drawing, and full 3D assembly. Updates live with sidebar inputs."
    )

    st.subheader("Blade Cross-Section")
    cs_col1, cs_col2 = st.columns([1, 3])
    with cs_col1:
        station_frac = st.slider(
            "Span station (r/R) for cross-section", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        thickness_pct = st.slider("Airfoil thickness (% chord)", min_value=6, max_value=18, value=12, step=1)
        station_r = root_cutout + station_frac * (radius - root_cutout)
        st.metric("Chord at this station", f"{rotor.get_chord(station_r)*1000:.1f} mm")
        st.metric("Twist at this station", f"{np.degrees(rotor.get_twist(station_r)):.2f}°")
    with cs_col2:
        st.plotly_chart(
            cross_section_figure(
                rotor.get_chord(station_r),
                thickness_ratio=thickness_pct / 100,
                station_label=f"Section at r/R = {station_frac:.2f}",
            ),
            width='stretch',
        )

    st.divider()
    st.subheader("Wing Chord & Twist Drawing")
    st.plotly_chart(engineering_drawing_figure(rotor), width='stretch')

    st.divider()
    st.subheader("3D Rotor Model")
    d1, d2 = st.columns([1, 3])
    with d1:
        n_span_pts = st.slider("3D mesh resolution (span stations)", min_value=8, max_value=48, value=24, step=4)
        st.caption("Drag to rotate • scroll to zoom • all blades shown around the hub.")

        st.markdown("**View presets**")
        if "rotor3d_view" not in st.session_state:
            st.session_state.rotor3d_view = "iso"

        vb1, vb2 = st.columns(2)
        with vb1:
            if st.button("⤢ Isometric", width='stretch'):
                st.session_state.rotor3d_view = "iso"
            if st.button("⬒ Top", width='stretch', help="Looking down the hub axis — best for seeing blade planform / sweep"):
                st.session_state.rotor3d_view = "top"
        with vb2:
            if st.button("▭ Front", width='stretch', help="Looking along the rotor's flight/drag direction — best for seeing coning and twist along span"):
                st.session_state.rotor3d_view = "front"
            if st.button("▯ Side", width='stretch', help="Looking along a blade's span — best for seeing twist angle change and airfoil orientation per station"):
                st.session_state.rotor3d_view = "side"

    with d2:
        # Camera presets (Plotly 3D "eye" vectors). Distances tuned for a rotor
        # that's wide in X/Y (span) and thin in Z (thickness), so twist is
        # readable from Front/Side without being too zoomed in or out.
        camera_presets = {
            "iso": dict(eye=dict(x=1.4, y=1.4, z=1.1), up=dict(x=0, y=0, z=1)),
            "top": dict(eye=dict(x=0.0, y=0.0, z=2.6), up=dict(x=0, y=1, z=0)),
            "front": dict(eye=dict(x=2.6, y=0.0, z=0.0), up=dict(x=0, y=0, z=1)),
            "side": dict(eye=dict(x=0.0, y=2.6, z=0.0), up=dict(x=0, y=0, z=1)),
        }
        camera = camera_presets[st.session_state.rotor3d_view]

        fig_3d = rotor_3d_figure(rotor, thickness_ratio=thickness_pct / 100, n_span=n_span_pts)
        fig_3d.update_layout(scene_camera=camera)
        st.plotly_chart(fig_3d, width='stretch')

with tab_overview:
    st.caption(
        "Preview of the blade geometry and airfoil aerodynamic model based on current sidebar inputs "
        "— updates live, no solve required."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_planform_figure(rotor), width='stretch')
    with c2:
        st.plotly_chart(make_polar_figure(rotor), width='stretch')

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Blades", f"{n_blades}")
    g2.metric("Radius", f"{radius:.3f} m")
    g3.metric("Disk Area", f"{np.pi * radius**2:.3f} m²")
    solidity = (n_blades * 0.5 * (chord_root + chord_tip)) / (np.pi * radius)
    g4.metric("Solidity (approx)", f"{solidity:.4f}")

with tab_results:
    if not run:
        st.info("Set your parameters in the sidebar and click **Run Solver** to see performance results.")
    else:
        rho, T_atm, p, a_sound = get_atmosphere(altitude_m, delta_T_ISA)

        st.subheader("Atmospheric Conditions")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Density (ρ)", f"{rho:.4f} kg/m³")
        col2.metric("Temperature (T)", f"{T_atm:.2f} K")
        col3.metric("Pressure (p)", f"{p:.0f} Pa")
        col4.metric("Speed of Sound (a)", f"{a_sound:.2f} m/s")

        with st.spinner("Running BEMT solver..."):
            results = solve_bemt(
                rotor, collective_rad, rpm, v_climb, rho, a_sound,
                n_elements=n_elements, tip_loss=tip_loss,
            )

        st.subheader("Performance Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Thrust (T)", f"{results['T']:.2f} N")
        col2.metric("Torque (Q)", f"{results['Q']:.2f} N·m")
        col3.metric("Power (P)", f"{results['P']/1000:.2f} kW")

        col4, col5, col6, col7 = st.columns(4)
        col4.metric("Thrust Coeff (CT)", f"{results['CT']:.6f}")
        col5.metric("Torque Coeff (CQ)", f"{results['CQ']:.6f}")
        col6.metric("Power Coeff (CP)", f"{results['CP']:.6f}")
        if results["FM"] > 0:
            col7.metric("Figure of Merit", f"{results['FM']:.4f}")
        else:
            col7.metric("Figure of Merit", "N/A (Not Hover)")

        df = pd.DataFrame({
            "r/R": results["r"] / radius,
            "dT (N)": results["dT"],
            "dQ (N-m)": results["dQ"],
            "Inflow Ratio (lambda)": results["lambda"],
            "Angle of Attack (deg)": np.degrees(results["alpha"]),
            "Inflow Angle (deg)": np.degrees(results["phi"]),
        })

        st.subheader("Radial Distributions")
        st.plotly_chart(make_radial_distributions_figure(df), width='stretch')
        st.plotly_chart(make_combined_loading_figure(df), width='stretch')

        st.subheader("Detailed Sectional Data")
        st.dataframe(df, width='stretch')
        st.download_button(
            "⬇ Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bemt_results.csv",
            mime="text/csv",
        )

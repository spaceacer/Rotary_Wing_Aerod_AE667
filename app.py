import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bemt.atmosphere import get_atmosphere
from bemt.rotor import Rotor
from bemt.solver import solve_bemt

st.set_page_config(page_title="BEMT Tool", layout="wide")

st.title("Blade Element Momentum Theory (BEMT) Solver")
st.markdown("Developed for Tiltrotor Helicopter Course Project - Milestone 1")

# Custom CSS to make the Run Solver button sticky at the bottom of the sidebar
st.markdown(
    """
    <style>
    /* Target the button container inside the sidebar */
    section[data-testid="stSidebar"] div.stButton {
        position: sticky;
        bottom: 0px;
        padding-bottom: 20px;
        padding-top: 10px;
        z-index: 999;
        /* Matches dark theme background to avoid transparency overlapping */
        background-color: rgb(38, 39, 48); 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs
st.sidebar.header("Rotor Geometry")
n_blades = st.sidebar.number_input("Number of Blades", min_value=2, max_value=8, value=2, step=1)
radius = st.sidebar.number_input("Rotor Radius (m)", min_value=0.1, value=0.762, step=0.1)
root_cutout = st.sidebar.number_input("Root Cut-out (m)", min_value=0.0, max_value=radius*0.9, value=0.125, step=0.01)
chord_root = st.sidebar.number_input("Root Chord (m)", min_value=0.01, value=0.0508, step=0.01)
chord_tip = st.sidebar.number_input("Tip Chord (m)", min_value=0.01, value=0.0508, step=0.01)
theta_root_deg = st.sidebar.number_input("Root Twist (deg)", value=0.0, step=1.0)
theta_tip_deg = st.sidebar.number_input("Tip Twist (deg)", value=0.0, step=1.0)

st.sidebar.header("Operating Conditions")
collective_deg = st.sidebar.number_input("Collective Pitch at 75%R (deg)", value=8.0, step=1.0)
rpm = st.sidebar.number_input("Rotational Speed (RPM)", min_value=100.0, value=2000.0, step=100.0)
v_climb = st.sidebar.number_input("Axial Velocity (m/s) (Climb/Forward)", value=0.0, step=1.0)

st.sidebar.header("Atmosphere")
altitude_m = st.sidebar.number_input("Altitude (m)", value=0.0, step=100.0)
delta_T_ISA = st.sidebar.number_input("ISA Temp Offset (K)", value=0.0, step=1.0)

st.sidebar.header("Solver Settings")
n_elements = st.sidebar.number_input("Number of Blade Elements", min_value=10, value=50, step=10)
tip_loss = st.sidebar.checkbox("Include Prandtl Tip Loss", value=True)

# Process inputs
theta_root = np.radians(theta_root_deg)
theta_tip = np.radians(theta_tip_deg)
collective_rad = np.radians(collective_deg)

if st.sidebar.button("Run Solver", type="primary"):
    # Atmosphere
    rho, T, p, a_sound = get_atmosphere(altitude_m, delta_T_ISA)
    
    st.header("Atmospheric Conditions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Density (rho)", f"{rho:.4f} kg/m^3")
    col2.metric("Temperature (T)", f"{T:.2f} K")
    col3.metric("Pressure (p)", f"{p:.0f} Pa")
    col4.metric("Speed of Sound (a)", f"{a_sound:.2f} m/s")
    
    # Rotor
    rotor = Rotor(n_blades, radius, root_cutout, chord_root, chord_tip, theta_root, theta_tip)
    
    # Solve
    with st.spinner("Running BEMT Solver..."):
        results = solve_bemt(rotor, collective_rad, rpm, v_climb, rho, a_sound, n_elements=n_elements, tip_loss=tip_loss)
    
    st.header("Performance Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Thrust (T)", f"{results['T']:.2f} N")
    col2.metric("Torque (Q)", f"{results['Q']:.2f} N-m")
    col3.metric("Power (P)", f"{results['P']/1000:.2f} kW")
    
    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Thrust Coeff (CT)", f"{results['CT']:.6f}")
    col5.metric("Torque Coeff (CQ)", f"{results['CQ']:.6f}")
    col6.metric("Power Coeff (CP)", f"{results['CP']:.6f}")
    if results['FM'] > 0:
        col7.metric("Figure of Merit", f"{results['FM']:.4f}")
    else:
        col7.metric("Figure of Merit", "N/A (Not Hover)")
        
    st.header("Radial Distributions")
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'r/R': results['r'] / radius,
        'dT (N)': results['dT'],
        'dQ (N-m)': results['dQ'],
        'Inflow Ratio (lambda)': results['lambda'],
        'Angle of Attack (deg)': np.degrees(results['alpha']),
        'Inflow Angle (deg)': np.degrees(results['phi'])
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(df['r/R'], df['dT (N)'], 'b-')
    axs[0, 0].set_title('Sectional Thrust')
    axs[0, 0].set_xlabel('r/R')
    axs[0, 0].set_ylabel('dT (N)')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(df['r/R'], df['dQ (N-m)'], 'r-')
    axs[0, 1].set_title('Sectional Torque')
    axs[0, 1].set_xlabel('r/R')
    axs[0, 1].set_ylabel('dQ (N-m)')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(df['r/R'], df['Inflow Ratio (lambda)'], 'g-')
    axs[1, 0].set_title('Inflow Ratio ($\lambda$)')
    axs[1, 0].set_xlabel('r/R')
    axs[1, 0].set_ylabel('$\lambda$')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(df['r/R'], df['Angle of Attack (deg)'], 'm-')
    axs[1, 1].set_title('Angle of Attack')
    axs[1, 1].set_xlabel('r/R')
    axs[1, 1].set_ylabel('Alpha (deg)')
    axs[1, 1].grid(True)
    
    st.pyplot(fig)
    
    st.header("Detailed Sectional Data")
    st.dataframe(df)

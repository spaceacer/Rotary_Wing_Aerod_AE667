"""
app.py
------
BEMT Rotor Analysis Tool — Streamlit entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

# Define the pages explicitly
dashboard = st.Page(
    "pages/2_BEMT_Dashboard.py", 
    title="BEMT Dashboard", 
    icon="🚁", 
    default=True
)
explorer = st.Page(
    "pages/1_Airfoil_Explorer.py", 
    title="Airfoil Explorer", 
    icon="✈️"
)
validation = st.Page(
    "pages/3_Validation.py", 
    title="Validation", 
    icon="📊"
)
mission_planner = st.Page(
    "pages/4_Mission_Planner.py",
    title="Mission Planner",
    icon="🗺️"
)

# Setup navigation
pg = st.navigation([dashboard, explorer, validation, mission_planner])

# Run the selected page
pg.run()

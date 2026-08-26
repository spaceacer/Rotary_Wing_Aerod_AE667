import re

with open('pages/2_BEMT_Dashboard.py', 'r') as f:
    content = f.read()

# We want to split the content right before `# ── build figure`
split_marker = "# ── build figure"
parts = content.split(split_marker)

if len(parts) != 2:
    print("Could not find split marker")
    exit(1)

head = parts[0]
tail = split_marker + parts[1]

# Indent the tail by 4 spaces
indented_tail = "\n".join(["    " + line if line else line for line in tail.split("\n")])

new_content = head + """
tab_perf, tab_inspect = st.tabs(["🚁 Rotor Performance", "🔍 Station Inspector"])

with tab_perf:
""" + indented_tail + """

with tab_inspect:
    st.subheader("Spanwise Station Inspector")
    
    min_rnorm = float(r_rc / radius)
    inspect_r = st.slider("Select radial station (r/R)", min_value=min_rnorm, max_value=1.0, value=min_rnorm + (1.0 - min_rnorm)/2, step=0.01)
    
    # Interpolate local properties from solver
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
    
    # 2D Blended Shape
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
    
    # Local Aerodynamic Polars
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
"""

with open('pages/2_BEMT_Dashboard.py', 'w') as f:
    f.write(new_content)

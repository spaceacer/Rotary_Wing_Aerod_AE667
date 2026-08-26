import re

with open('pages/2_BEMT_Dashboard.py', 'r') as f:
    content = f.read()

if 'import plotly.graph_objects as go' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport plotly.graph_objects as go')

split_marker = "# ── build figure"
parts = content.split(split_marker)
head = parts[0]
tail = split_marker + parts[1]

inspect_marker = "with tab_inspect:"
tail_parts = tail.split(inspect_marker)
tail_after_inspect = inspect_marker + tail_parts[1]

new_middle = """
tab_perf, tab_inspect = st.tabs(["🚁 Rotor Performance", "🔍 Station Inspector"])

with tab_perf:
    fig_2d = plt.figure(figsize=(7, 8))
    gs = fig_2d.add_gridspec(
        4, 1, hspace=0.48, left=0.08, right=0.95, top=0.95, bottom=0.06,
    )

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

    title_mode = "Mean Camber Line" if use_camber else "Lofted Blended 3-D Blade"
    af_desc = " → ".join(af_names)

    fig_3d = go.Figure(data=[go.Surface(
        x=Y_grid, y=X_grid, z=Z_grid, 
        surfacecolor=C_grid,
        colorscale=[[0, 'rgb(38,178,228)'], [1, 'rgb(255,64,64)']],
        showscale=False,
        cmin=0, cmax=1,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.1, specular=0.5, fresnel=0.2)
    )])

    max_c = max(c_root, c_root * taper)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=1.8 * np.cos(np.radians(st.session_state["azim"])) * np.cos(np.radians(st.session_state["elev"])),
            y=1.8 * np.sin(np.radians(st.session_state["azim"])) * np.cos(np.radians(st.session_state["elev"])),
            z=1.8 * np.sin(np.radians(st.session_state["elev"]))
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
            zaxis=dict(range=[-max_c, max_c]),
            aspectratio=dict(x=2.5, y=1.0, z=1.0),
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        height=650
    )

    col_3d, col_2d = st.columns([1.5, 1.0])
    with col_3d:
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_2d:
        st.pyplot(fig_2d, use_container_width=True)
        plt.close(fig_2d)

"""

new_content = head + split_marker + "\n" + new_middle + "\n" + tail_after_inspect

with open('pages/2_BEMT_Dashboard.py', 'w') as f:
    f.write(new_content)

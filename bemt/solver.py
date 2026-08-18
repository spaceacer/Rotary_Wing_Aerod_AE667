import numpy as np

def solve_bemt(rotor, collective_rad, rpm, v_climb, rho, a_sound, n_elements=50, tip_loss=True):
    omega = rpm * 2 * np.pi / 60.0
    R = rotor.radius
    
    r_arr = np.linspace(rotor.root_cutout, rotor.radius, n_elements + 1)
    r_centers = 0.5 * (r_arr[:-1] + r_arr[1:])
    dr = r_arr[1:] - r_arr[:-1]
    
    dT = np.zeros(n_elements)
    dQ = np.zeros(n_elements)
    lambda_arr = np.zeros(n_elements)
    alpha_arr = np.zeros(n_elements)
    phi_arr = np.zeros(n_elements)
    
    V_c = v_climb
    v_ti = omega * R
    
    for i, r in enumerate(r_centers):
        c = rotor.get_chord(r)
        theta_tw = rotor.get_twist(r)
        theta = collective_rad + theta_tw
        sigma_r = (rotor.n_blades * c) / (np.pi * R)
        
        lambda_c = V_c / v_ti
        lambda_i = 0.05 # initial guess
        
        tol = 1e-5
        max_iter = 500
        
        for it in range(max_iter):
            phi = np.arctan2(V_c + lambda_i * v_ti, omega * r)
            
            if tip_loss:
                # Prandtl tip loss
                f = (rotor.n_blades / 2) * (1 - r / R) / max(np.sin(phi), 1e-4)
                F = (2 / np.pi) * np.arccos(np.exp(-min(f, 20)))
                F = max(F, 1e-4)
            else:
                F = 1.0
                
            alpha = theta - phi
            cl, cd = rotor.get_cl_cd(alpha)
            
            U_T = omega * r
            U_P = V_c + lambda_i * v_ti
            U_sq = U_T**2 + U_P**2
            dL = 0.5 * rho * U_sq * c * cl * dr[i]
            dD = 0.5 * rho * U_sq * c * cd * dr[i]
            
            dT_element = rotor.n_blades * (dL * np.cos(phi) - dD * np.sin(phi))
            
            # Momentum theory inflow
            a_quad = 4 * np.pi * r * rho * v_ti**2 * F * dr[i]
            b_quad = 4 * np.pi * r * rho * V_c * v_ti * F * dr[i]
            c_quad = -dT_element
            
            discriminant = b_quad**2 - 4 * a_quad * c_quad
            if discriminant >= 0 and a_quad > 0:
                new_lambda_i = (-b_quad + np.sqrt(discriminant)) / (2 * a_quad)
            else:
                new_lambda_i = lambda_i # Keep same if non-physical
                
            lambda_i = 0.2 * new_lambda_i + 0.8 * lambda_i # Relaxation
            
            if np.abs(lambda_i - new_lambda_i) < tol:
                break
                
        # Final values
        phi = np.arctan2(V_c + lambda_i * v_ti, omega * r)
        alpha = theta - phi
        cl, cd = rotor.get_cl_cd(alpha)
        
        U_T = omega * r
        U_P = V_c + lambda_i * v_ti
        U_sq = U_T**2 + U_P**2
        
        dL = 0.5 * rho * U_sq * c * cl * dr[i]
        dD = 0.5 * rho * U_sq * c * cd * dr[i]
        
        dT[i] = rotor.n_blades * (dL * np.cos(phi) - dD * np.sin(phi))
        dQ[i] = rotor.n_blades * (dL * np.sin(phi) + dD * np.cos(phi)) * r
        
        lambda_arr[i] = lambda_i + V_c / v_ti
        alpha_arr[i] = alpha
        phi_arr[i] = phi
        
    T = np.sum(dT)
    Q = np.sum(dQ)
    P = Q * omega
    
    # Non-dimensional coefficients
    A = np.pi * R**2
    CT = T / (rho * A * v_ti**2)
    CQ = Q / (rho * A * v_ti**2 * R)
    CP = P / (rho * A * v_ti**3)
    
    # Figure of merit (Hover only)
    FM = 0
    if V_c == 0 and CP > 0:
        CP_ideal = (CT**1.5) / np.sqrt(2)
        FM = CP_ideal / CP
        
    return {
        'T': T,
        'Q': Q,
        'P': P,
        'CT': CT,
        'CQ': CQ,
        'CP': CP,
        'FM': FM,
        'r': r_centers,
        'dT': dT,
        'dQ': dQ,
        'lambda': lambda_arr,
        'alpha': alpha_arr,
        'phi': phi_arr
    }

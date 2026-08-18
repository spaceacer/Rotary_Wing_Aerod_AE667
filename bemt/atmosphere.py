import numpy as np

def get_atmosphere(altitude_m, delta_T_ISA=0.0):
    """
    Computes atmospheric properties according to the International Standard Atmosphere (ISA).
    
    Parameters:
    - altitude_m: Altitude in meters
    - delta_T_ISA: Temperature offset from ISA in Kelvin (or Celsius)
    
    Returns:
    - rho: Air density (kg/m^3)
    - T: Temperature (K)
    - p: Pressure (Pa)
    - a: Speed of sound (m/s)
    """
    # Sea level standard conditions
    T0 = 288.15 # K
    p0 = 101325.0 # Pa
    g = 9.80665 # m/s^2
    R = 287.05 # J/(kg.K)
    L = 0.0065 # Temperature lapse rate (K/m) in troposphere
    gamma = 1.4
    
    if altitude_m <= 11000:
        # Troposphere
        T_isa = T0 - L * altitude_m
        p = p0 * (1 - L * altitude_m / T0) ** (g / (R * L))
    else:
        # Lower Stratosphere (simplified, up to ~20km)
        T11 = T0 - L * 11000
        p11 = p0 * (1 - L * 11000 / T0) ** (g / (R * L))
        T_isa = T11
        p = p11 * np.exp(-g / (R * T11) * (altitude_m - 11000))
        
    # Apply temperature offset
    T = T_isa + delta_T_ISA
    
    # Calculate density using ideal gas law
    rho = p / (R * T)
    
    # Speed of sound
    a = np.sqrt(gamma * R * T)
    
    return rho, T, p, a

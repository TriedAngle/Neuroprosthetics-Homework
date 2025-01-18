import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

N_COMPARTMENTS = 100
DT = 25e-6  # 25 µs
T_END = 0.03  # 30 ms
T = 6.3  # Temperature in °C

def calculate_extracellular_potential(x: np.ndarray, y: float, I: float, t: float,
                                    stim_start: float, phase_duration: float,
                                    biphasic: bool = False) -> np.ndarray:
    """Calculate extracellular potential at given positions and time."""
    rho = 1.0  # Ω·m
    electrode_x = 150e-6  
    r = np.sqrt((x - electrode_x)**2 + y**2)
    
    if t < stim_start or t > stim_start + phase_duration * (2 if biphasic else 1):
        return np.zeros_like(x)
    elif biphasic and t > stim_start + phase_duration:
        I = -I 
    
    return (rho * I) / (4 * np.pi * r)

def calculate_axonal_resistance(rho_axon: float, r_axon: float, l_comp: float) -> float:
    """
    Calculate axonal resistance between compartments.
    
    Inputs:
        rho_axon: Axonal resistivity (Ω·m)
        r_axon: Axon radius (m)
        l_comp: Compartment length (m)
    
    Outputs:
        float: Axonal resistance (Ω)
    """
    return (rho_axon * l_comp) / (np.pi * r_axon**2)

def calculate_rate_constants(V: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Calculate rate constants for the H-H model gates.
    
    Inputs:
        V: Membrane potential array (V)
    
    Outputs:
        Tuple of rate constants (alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h)
    """
    alpha_m = 1000 * (2.5 - 0.1*V*1000) / (np.exp(2.5 - 0.1*V*1000) - 1)
    beta_m = 4000 * np.exp(-V*1000/18)
    
    alpha_n = 100 * (1 - 0.1*V*1000) / (np.exp(1 - 0.1*V*1000) - 1)
    beta_n = 125 * np.exp(-V*1000/80)
    
    alpha_h = 70 * np.exp(-V*1000/20)
    beta_h = 1000 / (np.exp(3 - 0.1*V*1000) + 1)
    
    return alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h

def calculate_ionic_currents(V: np.ndarray, m: np.ndarray, n: np.ndarray, 
                           h: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Calculate ionic currents for the H-H model.
    
    Inputs:
        V: Membrane potential array (V)
        m, n, h: Gating variables arrays
    
    Outputs:
        ionic currents (i_na, i_k, i_l)
    """
    g_Na = 120e-3  # Sodium conductance (S)
    g_K = 36e-3    # Potassium conductance (S)
    g_L = 0.3e-3   # Leak conductance (S)
    E_Na = 115e-3  # V
    E_K = -12e-3   # V
    E_L = 10.6e-3  # V

    i_na = g_Na * m**3 * h * (V - E_Na)
    i_k = g_K * n**4 * (V - E_K)
    i_l = g_L * (V - E_L)
    
    return i_na, i_k, i_l

def hh_gating(V: np.ndarray, dt: float, curr_gate: np.ndarray, T: float) -> np.ndarray:
    """
    Calculate gating variables for the next time step.
    
    Inputs:
        V: Membrane potential array (V)
        dt: Time step (s)
        curr_gate: Current gating variables array (3xN)
        T: Temperature (°C)
    
    Outputs:
        Updated gating variables (3xN)
    """
    alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h = calculate_rate_constants(V)
    
    k = 3.0**(0.1 * (T - 6.3))
    
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_m = 1.0 / ((alpha_m + beta_m) * k)
    
    n_inf = alpha_n / (alpha_n + beta_n)
    tau_n = 1.0 / ((alpha_n + beta_n) * k)
    
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_h = 1.0 / ((alpha_h + beta_h) * k)
    
    m_new = m_inf + (curr_gate[0] - m_inf) * np.exp(-dt/tau_m)
    n_new = n_inf + (curr_gate[1] - n_inf) * np.exp(-dt/tau_n)
    h_new = h_inf + (curr_gate[2] - h_inf) * np.exp(-dt/tau_h)
    
    return np.array([m_new, n_new, h_new])

def hh_potential(V: np.ndarray, Ve: np.ndarray, dt: float, I_ions: np.ndarray, 
                Ra: float, Cm: float) -> np.ndarray:
    """
    Calculate membrane potential for the next time step using implicit Euler.
    
    Inputs:
        V: Current membrane potential array (V)
        dt: Time step (s)
        I_ions: Ionic currents array
        I_stim: Stimulation current array
        Ra: Axonal resistance (Ω)
        Cm: Membrane capacitance (F)
    
    Outputs:
        Updated membrane potential array
    """
    n = len(V)
    diag = np.ones(n) * (1 + 2*dt/(Ra*Cm))
    off_diag = np.ones(n-1) * (-dt/(Ra*Cm))
    
    A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    A[0,1] = -2*dt/(Ra*Cm)
    A[-1,-2] = -2*dt/(Ra*Cm)
    
    dVe = np.zeros_like(V)
    dVe[1:-1] = (Ve[2:] - 2*Ve[1:-1] + Ve[:-2])
    
    b = V + (dt/Cm) * (-I_ions + dVe/(Ra))

    V_new = np.linalg.solve(A, b)
    
    return V_new

def run_simulation(I_stim: float, biphasic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    rho_axon = 7e-3  # Ω·m
    r_axon = 1e-6    # m
    l_comp = 3e-6    # m
    Cm = 1e-6        # F/cm²
    y_dist = 10e-6   # m (distance from electrode)
    
    Ra = calculate_axonal_resistance(rho_axon, r_axon, l_comp)
    
    t = np.arange(0, T_END, DT)
    x = np.linspace(0, 300e-6, N_COMPARTMENTS)  # 300 µm axon
    
    V = np.zeros((len(t), N_COMPARTMENTS))
    Ve = np.zeros((len(t), N_COMPARTMENTS))
    gates = np.zeros((3, len(t), N_COMPARTMENTS))
    
    alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h = calculate_rate_constants(V[0])
    gates[0,0] = alpha_m/(alpha_m + beta_m)
    gates[1,0] = alpha_n/(alpha_n + beta_n)
    gates[2,0] = alpha_h/(alpha_h + beta_h)
    
    # Main simulation loop
    for i in range(len(t)-1):
        Ve[i] = calculate_extracellular_potential(
            x, y_dist, I_stim, t[i], 0.005, 0.001, biphasic)
        
        I_na, I_k, I_l = calculate_ionic_currents(V[i], gates[0,i], gates[1,i], gates[2,i])
        I_total = I_na + I_k + I_l
        
        V[i+1] = hh_potential(V[i], Ve[i], DT, I_total, Ra, Cm)
        gates[:,i+1] = hh_gating(V[i], DT, gates[:,i], T)
    
    return t, V

def plot_results(t: np.ndarray, V: np.ndarray, title: str, filename: str):
    plt.figure(figsize=(16, 6))
    plt.pcolormesh(t*1000, np.arange(N_COMPARTMENTS)+1, V.T*1000, 
                   shading='auto', cmap='viridis', vmin=-20, vmax=100)
    
    cbar = plt.colorbar(label='V (mV)')
    plt.setp(cbar.ax.get_yticklabels(), fontsize=12)
    cbar.set_label('V (mV)', fontsize=14)
    
    plt.xlabel('t (ms)', fontsize=14)
    plt.ylabel('Compartment Nr.', fontsize=14)
    
    plt.title(title, fontsize=16, pad=20)
    
    plt.xticks(np.arange(0, 31, 5), fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.gca().invert_yaxis()
    plt.xlim(0, 30)
    
    os.makedirs('out', exist_ok=True)
    plt.savefig(os.path.join('out', filename))
    plt.close()

def main():
    params = [
        (-0.05e-3, False, "Mono-phasic -0.05mA"),
        (-0.1e-3, False, "Mono-phasic -0.1mA"),
        (-0.1e-3, True, "Bi-phasic ∓0.1mA"),
        (-0.15e-3, True, "Bi-phasic ∓0.15mA"),
        (0.2e-3, False, "Mono-phasic 0.2mA"),
        (0.4e-3, False, "Mono-phasic 0.4mA")
    ]
    
    for current, is_biphasic, description in params:
        current_str = f"{abs(current*1e3):.2f}mA"
        stim_type = "biphasic" if is_biphasic else "monophasic"
        polarity = "neg" if current < 0 else "pos"
        filename = f"ap_propagation_{stim_type}_{polarity}_{current_str}.png"
        
        t, V = run_simulation(current, is_biphasic)
        plot_results(t, V, 
                    f"Action Potential Propagation\n{description}",
                    filename)

if __name__ == "__main__":
    main()

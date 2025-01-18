import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

N_COMPARTMENTS = 100
SAVE_FIGS = True
SAVE_PNG = True


def calculate_axonal_resistance(
    rho_axon: float,
    r_axon: float,
    l_comp: float
) -> float:
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


# had this split before, but I discovered Tuple[...] and thought it would be cool
def calculate_rate_constants(V: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Calculate rate constants for the H-H model gates.
    
    Inputs:
        V: Membrane potential array (V)
    
    Outputs:
        Tuple of rate constants (alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h)
    """
    alpha_m = 1000 * (2.5 - 100*V) / (np.exp(2.5 - 100*V) - 1)
    beta_m = 4000 * np.exp(-500*V/9)
    
    alpha_n = 1000 * (0.1 - 10*V) / (np.exp(1 - 100*V) - 1)
    beta_n = 125 * np.exp(-25*V/2)
    
    alpha_h = 70 * np.exp(-50*V)
    beta_h = 1000 / (np.exp(3 - 100*V) + 1)
    
    return alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h


def calculate_ionic_currents(
    V: np.ndarray,
    m: np.ndarray,
    n: np.ndarray,
    h: np.ndarray
) -> Tuple[np.ndarray, ...]:
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
    V_Na = 115e-3  # Sodium Nernst potential (V)
    V_K = -12e-3   # Potassium Nernst potential (V)
    V_L = 10.6e-3  # Leak Nernst potential (V)
    
    i_na = g_Na * m**3 * h * (V - V_Na)
    i_k = g_K * n**4 * (V - V_K)
    i_l = g_L * (V - V_L)
    
    return i_na, i_k, i_l

def hh_gating(
    V: np.ndarray,
    dt: float,
    curr_gate: np.ndarray,
    T: float
) -> np.ndarray:
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
    
    # Temperature correction
    k = 3.0**(0.1 * (T - 6.3))
    
    A_m = alpha_m / (alpha_m + beta_m)
    B_m = 1.0 / ((alpha_m + beta_m) * k)
    
    A_n = alpha_n / (alpha_n + beta_n)
    B_n = 1.0 / ((alpha_n + beta_n) * k)
    
    A_h = alpha_h / (alpha_h + beta_h)
    B_h = 1.0 / ((alpha_h + beta_h) * k)
    
    m_new = A_m + (curr_gate[0] - A_m) * np.exp(-dt/B_m)
    n_new = A_n + (curr_gate[1] - A_n) * np.exp(-dt/B_n)
    h_new = A_h + (curr_gate[2] - A_h) * np.exp(-dt/B_h)
    
    return np.array([m_new, n_new, h_new])

def hh_potential(
    V: np.ndarray,
    dt: float,
    I_ions: np.ndarray,
    I_stim: np.ndarray,
    Ra: float,
    Cm: float
) -> np.ndarray:
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
    # tridiagonal matrix for implicit solver
    n = len(V)
    diag = np.ones(n) * (1 + 2*dt/(Ra*Cm))
    off_diag = np.ones(n-1) * (-dt/(Ra*Cm))
    
    A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # Boundary conditions (no current flow at ends)
    A[0,1] = -2*dt/(Ra*Cm)
    A[-1,-2] = -2*dt/(Ra*Cm)
    
    # right hand side
    b = V + (dt/Cm) * (-np.sum(I_ions, axis=0) + I_stim)
    
    V_new = np.linalg.solve(A, b)
    
    return V_new

def run_simulation(
    t_end: float,
    dt: float,
    T: float,
    stim_times: List[float], 
    stim_compartments: List[int],
    stim_amplitudes: List[float],
    stim_duration: float
) -> Tuple[np.ndarray, ...]:
    """
    Run the multicompartment H-H model simulation.
    
    Inputs:
        t_end: Simulation duration (s)
        dt: Time step (s)
        T: Temperature (°C)
        stim_times: List of stimulation start times (s)
        stim_compartments: List of compartments to stimulate
        stim_amplitudes: List of stimulation amplitudes (A)
        stim_duration: Stimulation duration (s)
    
    Outputs:
        Tuple of (time array, voltage array, gating variables array, ionic currents array)
    """
    Cm = 1e-6      # Membrane capacitance (F)
    rho_axon = 1.0 # Axonal resistivity (Ω·m)
    r_axon = 2e-6  # Axon radius (m)
    l_comp = 1e-7  # Compartment length (m)
    
    Ra = calculate_axonal_resistance(rho_axon, r_axon, l_comp)
    
    t = np.arange(0, t_end, dt)
    V = np.zeros((len(t), N_COMPARTMENTS))
    gates = np.zeros((3, len(t), N_COMPARTMENTS))
    I_ions = np.zeros((3, len(t), N_COMPARTMENTS))
    
    I_stim = np.zeros((len(t), N_COMPARTMENTS))
    for start_time, comp, amp in zip(stim_times, stim_compartments, stim_amplitudes):
        idx_start = int(start_time/dt)
        idx_end = int((start_time + stim_duration)/dt)
        I_stim[idx_start:idx_end, comp-1] = amp
    
    alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h = calculate_rate_constants(V[0])
    gates[0,0] = alpha_m/(alpha_m + beta_m)
    gates[1,0] = alpha_n/(alpha_n + beta_n)
    gates[2,0] = alpha_h/(alpha_h + beta_h)
    
    for i in range(len(t)-1):
        I_ions[:,i] = calculate_ionic_currents(V[i], gates[0,i], gates[1,i], gates[2,i])
        V[i+1] = hh_potential(V[i], dt, I_ions[:,i], I_stim[i], Ra, Cm)
        gates[:,i+1] = hh_gating(V[i], dt, gates[:,i], T)
    
    return t, V, gates, I_ions

def plot_results(t: np.ndarray, V: np.ndarray, name: str, stim_times: List[float], stim_compartments: List[int]):
    plt.figure(figsize=(16, 8))
    
    plt.pcolormesh(t*1000, np.arange(N_COMPARTMENTS)+1, V.T*1000, 
                   shading='auto', cmap='viridis')
    
    plt.colorbar(label='V (mV)')
    
    plt.xticks(np.arange(0, 101, 10))
    
    plt.xlabel('t (ms)', fontsize=18)
    plt.ylabel('Compartment Nr.', fontsize=18)
    
    plt.gca().invert_yaxis()
    
    stim_info = ", ".join([f"(t={t*1000:.1f}ms,comp={c})" for t, c in zip(stim_times, stim_compartments)])
    plt.title(f'Action Potential Propagation\nStimulation points: {stim_info}', fontsize=20)
    
    if SAVE_FIGS:
        output_dir = "out/svg" if not SAVE_PNG else "out/png"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.join(output_dir, f"ap_propagation_{name}.{'png' if SAVE_PNG else 'svg'}")
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    t_end = 0.1    # 100 ms
    dt = 25e-6     # 25 µs
    T = 6.3        # °C
    
    t1, V1, _, _ = run_simulation(
        t_end=t_end,
        dt=dt,
        T=T,
        stim_times=[0.005],
        stim_compartments=[N_COMPARTMENTS],
        stim_amplitudes=[5e-6],
        stim_duration=0.005
    )
    plot_results(t1, V1, "single_stim", [0.005], [N_COMPARTMENTS])
    
    t2, V2, _, _ = run_simulation(
        t_end=t_end,
        dt=dt,
        T=T,
        stim_times=[0.0, 0.015],
        stim_compartments=[20, 80],
        stim_amplitudes=[5e-6, 5e-6],
        stim_duration=0.005
    )
    plot_results(t2, V2, "double_stim", [0.0, 0.015], [20, 80])

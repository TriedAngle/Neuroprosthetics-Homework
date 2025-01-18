"""
------------------Exercise 4 hh_model_complete template-------------------------
   Term:    WS2024
   Date:    08.12.2024
   Name:    Sebastian Strobl
-------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import os

"""
This Part is used to define global lambda functions.
You can skip it for now as you will be sent back to this part later.
Start coding in line 38
"""
# Define Voltage ODE
# This has to be done in a global scope, as every lambda function acts as a function itself
# You can utilize anonymous (lambda) functions for this and the rate equations.
# Example: to declare a square function:
# sqr = lambda n: n**2
# To use the lambda function:
# x = sqr(n)
# Insert your code here:


# Define rate equations
#   alpha and beta equations for m,n,h 
# Insert your code here:


# we aren"t using any state in these "lambdas" anyways, so make defs (otherwise lsp mad haha).

def alpha_m(V):
    """Sodium activation rate alpha."""
    return 1000 * (2.5 - 100 * V) / (np.exp(2.5 - 100 * V) - 1)

def beta_m(V):
    """Sodium activation rate beta."""
    return 4000 * np.exp(-500 * V / 9)

def alpha_n(V):
    """Potassium activation rate alpha."""
    return 1000 * (0.1 - 10*V) / (np.exp(1 - 100 * V) - 1)

def beta_n(V):
    """Potassium activation rate beta."""
    return 125 * np.exp(-25 * V / 2)

def alpha_h(V):
    """Sodium inactivation rate alpha."""
    return 70 * np.exp(-50 * V)

def beta_h(V):
    """Sodium inactivation rate beta."""
    return 1000 / (np.exp(3 - 100 *V) + 1)


def hh_gating(V, dt, curr_gate, T):
    """
  this function calculates the gating variables for a future time step
    
    Inputs:
        V:          Membrane potential of the current time step in V
        dt:         time step in s
        curr_gate:  gating variables of the current time step (3x1 vector)
        T:          Simulation temperature in °C
        
    Outputs:
        new_gate:       gating variables of the next time step (vector 3x1)
    """
    ## 1) calculate gating variables
    
    # alpha and beta rate equations
	# Hint: you can also use "anonymous functions" for the rate equations, see
	# MATLAB documentation
    a_m, b_m = alpha_m(V), beta_m(V)
    a_n, b_n = alpha_n(V), beta_n(V)
    a_h, b_h = alpha_h(V), beta_h(V)
    
    # Temperature correction
    k = 3.0**(0.1 * (T - 6.3))

    # A and B coefficients for exponential solver
    A_m = a_m / (a_m + b_m)
    B_m = 1.0 / ((a_m + b_m) * k)
    
    A_n = a_n / (a_n + b_n)
    B_n = 1.0 / ((a_n + b_n) * k)
    
    A_h = a_h / (a_h + b_h)
    B_h = 1.0 / ((a_h + b_h) * k)
    
    # calculate gating variables m,n,h for future timestep using the exponential euler solver
    m_new = A_m + (curr_gate[0] - A_m) * np.exp(-dt/B_m)
    n_new = A_n + (curr_gate[1] - A_n) * np.exp(-dt/B_n)
    h_new = A_h + (curr_gate[2] - A_h) * np.exp(-dt/B_h)
    
    ## 2) assign output
    return np.array([m_new, n_new, h_new])


def hh_potential(V, dt, I_ions, I_stim):
    """
  this function calculates the membrane potential for a future time step
 
    Inputs:
        V:          Membrane potential of the current time step in V
        dt:         time step in s
        I_ions:     ionic currents of a current timestep (3x1 vector)
        I_stim:     stimulation current of a current timestep
 
    Outputs:
        V_new:      Membrane potential of a future timestep
    """
    ## 1) calculate new membrane potential
	# parameters
    C_m = 1e-6
    
    # Add up the currents
    I_total = -np.sum(I_ions) + I_stim
    
    # calculate new membrane potential with forward Euler
    V_new = V + (dt/C_m) * I_total
    
    ## 2) assign output
    return V_new

def hh_model(I_stim, t_end, dt, T, V_rest: float = 0.0):
    """
    this function simulates a Hodgkin Huxley neuron model
 
    Inputs:
        I_stim:     stimulation current as a vector
        t_end:      simulation duration in s
        dt:         time step in s
        T:          simulation temperature in °C
        V_rest:     membrane resting potential in V
    Outputs:
        V:          membrane potentials as a vector
        gates:      gating variables in a 3xlength(t) matrix form, rows being m - n - h
        I_ions:     ion currents in a 3xlength(t) matrix form: rows being i_na - i_k - i_l
        t:          time vector
    """
   	
    ## Definitions and constants
    V_rest = V_rest # Resting potential (V)
    g_Na = 120e-3   # Sodium conductance (S)
    g_K = 36e-3     # Potassium conductance (S)
    g_L = 0.3e-3    # Leak conductance (S)
    V_Na = 115e-3   # Sodium Nernst potential (V)
    V_K = -12e-3    # Potassium Nernst potential (V)
    V_L = 10.6e-3   # Leak Nernst potential (V)
    
    t = np.arange(0, t_end, dt)
    
    # Potential vector 1xlength
    V = np.zeros(len(t))

    # all matrices are 3xlenght, with the rows always being m - n - h
    gates = np.zeros((3, len(t)))
    I_ions = np.zeros((3, len(t)))
    
    # Set initial conditions
    V[0] = V_rest
    
    # Calculate initial steady-state values for gates
    a_m, b_m = alpha_m(V[0]), beta_m(V[0])
    a_n, b_n = alpha_n(V[0]), beta_n(V[0])
    a_h, b_h = alpha_h(V[0]), beta_h(V[0])
    
    # use the steady-state equations to obtain the initial gating variable states
    gates[0,0] = a_m/(a_m + b_m)
    gates[1,0] = a_n/(a_n + b_n)
    gates[2,0] = a_h/(a_h + b_h)
    
	## iterative calculation of the membrane potential
    for i in range(len(t)-1):
		# calculate ionic currents for current timestep
        I_ions[0,i] = g_Na * gates[0,i]**3 * gates[2,i] * (V[i] - V_Na)
        I_ions[1,i] = g_K * gates[1,i]**4 * (V[i] - V_K)
        I_ions[2,i] = g_L * (V[i] - V_L)
        
		# calculate membrane potential of a future timestep
        V[i+1] = hh_potential(V[i], dt, I_ions[:,i], I_stim[i])
        
		# calculate gating variables of a future timestep
        gates[:,i+1] = hh_gating(V[i], dt, gates[:,i], T)
    
    ## assign outputs
    return V, gates, I_ions, t



SAVE_FIGS = True
SAVE_PNG = True

# 1) Define Variables
t_end   = 0.1	# Simulation duration in s
dt      = 1e-5	# Time step in s
T1      = 6.3	# Simulation temperature in °C case 1
T2      = 28.0	# Simulation temperature in °C case 2

#  2) Create two stimulation currents
t = np.arange(0, t_end, dt)
I_stim1 = np.zeros_like(t)
I_stim2 = np.zeros_like(t)

pulse_duration = 0.005
# unless I do this, numpy doesn"t render the rising flag
pulse_times = [0.00000000001, 0.015, 0.030, 0.045, 0.060]

amplitudes1 = [2e-6, 3e-6, 4e-6, 6e-6, 8e-6]
amplitudes2 = [2e-6, 4e-6, 8e-6, 16e-6, 32e-6]

for start_time, amp1, amp2 in zip(pulse_times, amplitudes1, amplitudes2):
    pulse_indices = (t >= start_time) & (t < start_time + pulse_duration)
    I_stim1[pulse_indices] = amp1
    I_stim2[pulse_indices] = amp2

#  3) Run the hh_model function for both stimuli (with different temperatures)
V1, gates1, I_ions1, t1 = hh_model(I_stim1, t_end, dt, T1)
V2, gates2, I_ions2, t2 = hh_model(I_stim2, t_end, dt, T2)


#  4) Plot the results
plots = {}
plt.rcParams.update({"font.size": 17})

# Insert your code here:
# Plot 1 a): Input current I_stim1 for simulation at 6.3°C
plt.figure(figsize=(10, 6))
plt.plot(t1*1000, I_stim1*1e6)
plt.title("Input Current (6.3°C)")
plt.xlabel("t (ms)")
plt.ylabel("I (µA)")
plt.grid(True)
plt.xlim(-1, t_end*1000)
plots["input_current_63"] = plt.gcf()

# Plot 1 b): Input current I_stim2 for simulation at 28°C
plt.figure(figsize=(10, 6))
plt.plot(t2*1000, I_stim2*1e6)
plt.title("Input Current (28°C)")
plt.xlabel("t (ms)")
plt.ylabel("I (µA)")
plt.grid(True)
plt.xlim(-1, t_end*1000)
plots["input_current_28"] = plt.gcf()


# Plot 2 a): Membrane Potential for I_stim1 and 6.3°C
plt.figure(figsize=(10, 6))
plt.plot(t1*1000, V1*1000)
plt.title("Membrane Potential (6.3°C)")
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.grid(True)
plots["membrane_potential_63"] = plt.gcf()

# Plot 2 b): Membrane Potential for I_stim2 and 28°C
plt.figure(figsize=(10, 6))
plt.plot(t2*1000, V2*1000)
plt.title("Membrane Potential (28°C)")
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.grid(True)
plots["membrane_potential_28"] = plt.gcf()


# Plot 3 a): Gating Variables for I_stim1 and 6.3°C
plt.figure(figsize=(10, 6))
plt.plot(t1*1000, gates1[0], label="m")
plt.plot(t1*1000, gates1[1], label="n")
plt.plot(t1*1000, gates1[2], label="h")
plt.title("Gating Variables (6.3°C)")
plt.xlabel("t (ms)")
plt.ylabel("opening probability")
plt.legend()
plt.grid(True)
plots["gating_variables_63"] = plt.gcf()

# Plot 3 b): Gating Variables for I_stim2 and 28°C
plt.figure(figsize=(10, 6))
plt.plot(t2*1000, gates2[0], label="m")
plt.plot(t2*1000, gates2[1], label="n")
plt.plot(t2*1000, gates2[2], label="h")
plt.title("Gating Variables (28°C)")
plt.xlabel("t (ms)")
plt.ylabel("opening probability")
plt.legend()
plt.grid(True)
plots["gating_variables_28"] = plt.gcf()

plt.figure(figsize=(10, 6))
plt.plot(t2*1000, gates2[0], label="m")
plt.plot(t2*1000, gates2[1], label="n")
plt.plot(t2*1000, gates2[2], label="h")
plt.title("Closeup Gating Variables (28°C)")
plt.xlabel("t (ms)")
plt.ylabel("opening probability")
plt.xlim(40, 70)
plt.legend()
plt.grid(True)
plots["gating_variables_closeup_28"] = plt.gcf()



# Plot 4 a): Currents I_Na and I_K for I_stim1 and 6.3°C
plt.figure(figsize=(10, 6))
plt.plot(t1*1000, I_ions1[0]*1e6, label="I_Na")
plt.plot(t1*1000, I_ions1[1]*1e6, label="I_K")
plt.title("Ionic Currents (6.3°C)")
plt.xlabel("t (ms)")
plt.ylabel("I (µA)")
plt.legend()
plt.grid(True)
plots["ionic_currents_63"] = plt.gcf()

# Plot 4 b): Currents I_Na and I_K for I_stim2 and 28°C
plt.figure(figsize=(10, 6))
plt.plot(t2*1000, I_ions2[0]*1e6, label="I_Na")
plt.plot(t2*1000, I_ions2[1]*1e6, label="I_K")
plt.title("Ionic Currents (28°C)")
plt.xlabel("t (ms)")
plt.ylabel("I (µA)")
plt.legend()
plt.grid(True)
plots["ionic_currents_28"] = plt.gcf()


# no idea if these plots are correct, 
# they were in the template but on in the exercise. just leaving them here
# Plot 5 a): Phase plot for I_stim1 and 6.3°C
plt.figure(figsize=(10, 6))
plt.plot(V1*1000, gates1[0])
plt.title("Phase Plot (6.3°C)")
plt.xlabel("Membrane Potential (mV)")
plt.ylabel("m-gate")
plt.grid(True)
plots["phase_plot_63"] = plt.gcf()

# Plot 5 b): Phase plot for I_stim2 and 28°C
plt.figure(figsize=(10, 6))
plt.plot(V2*1000, gates2[0])
plt.title("Phase Plot (28°C)")
plt.xlabel("Membrane Potential (mV)")
plt.ylabel("m-gate")
plt.grid(True)
plots["phase_plot_28"] = plt.gcf()

if not SAVE_FIGS:
    plt.show()

if SAVE_FIGS:
    output_dir = "out/svg"
    output_type = "svg"
    if SAVE_PNG:
        output_dir = "out/png"
        output_type = "png"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plot_name, figure in plots.items():
        filename = os.path.join(output_dir, f"{plot_name}.{output_type}")
        figure.savefig(filename)

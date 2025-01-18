import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_FIGS = True
SAVE_PNG = True

def calculate_rate_constants(V):
    alpha_m = 1000 * (2.5 - 100*V) / (np.exp(2.5 - 100*V) - 1)
    beta_m = 4000 * np.exp(-500*V/9)
    
    # n gate rate constants
    alpha_n = 1000 * (0.1 - 10*V) / (np.exp(1 - 100*V) - 1)
    beta_n = 125 * np.exp(-25*V/2)
    
    # h gate rate constants
    alpha_h = 70 * np.exp(-50*V)
    beta_h = 1000 / (np.exp(3 - 100*V) + 1)
    
    return alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h

def calculate_time_constants(alpha, beta, T):
    k = 3.0**(0.1*(T-6.3))
    return 1.0 / ((alpha + beta) * k)

def calculate_steady_state(alpha, beta):
    return alpha / (alpha + beta)

V = np.linspace(-0.1, 0.1, 1000)

temperatures = [6.3, 28]
plt.figure(figsize=(12, 8))

for idx, T in enumerate(temperatures):
    alpha_m, beta_m, alpha_n, beta_n, alpha_h, beta_h = calculate_rate_constants(V)
    
    tau_m = calculate_time_constants(alpha_m, beta_m, T)
    tau_n = calculate_time_constants(alpha_n, beta_n, T)
    tau_h = calculate_time_constants(alpha_h, beta_h, T)
    
    m_inf = calculate_steady_state(alpha_m, beta_m)
    n_inf = calculate_steady_state(alpha_n, beta_n)
    h_inf = calculate_steady_state(alpha_h, beta_h)
    
    plt.subplot(2, 2, idx+1)
    plt.plot(V*1000, tau_m*1000, label="τm")
    plt.plot(V*1000, tau_n*1000, label="τn")
    plt.plot(V*1000, tau_h*1000, label="τh")
    plt.xlabel("V (mV)")
    plt.ylabel("t (ms)")
    plt.title(f"Time Constants at {T}°C")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, idx+3)
    plt.plot(V*1000, m_inf, label="m∞")
    plt.plot(V*1000, n_inf, label="n∞")
    plt.plot(V*1000, h_inf, label="h∞")
    plt.xlabel("V (mV)")
    plt.ylabel("Steady State Value")
    plt.title(f"Steady States at {T}°C")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
if not SAVE_FIGS:
    plt.show()

if SAVE_FIGS:
    figure = plt.gcf()

    output_dir = "out/svg"
    output_type = "svg"
    plot_name = "ion_channels"
    if SAVE_PNG:
        output_dir = "out/png"
        output_type = "png"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"{plot_name}.{output_type}")
    figure.savefig(filename)


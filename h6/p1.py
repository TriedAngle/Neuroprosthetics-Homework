import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_potential(x: np.ndarray, y: float, I: float, rho: float = 1.0) -> np.ndarray:
    r = np.sqrt((x-150e-6)**2 + y**2)
    return (rho * I) / (4 * np.pi * r)

def calculate_efield(x: np.ndarray, y: float, I: float, rho: float = 1.0) -> np.ndarray:
    r = np.sqrt((x-150e-6)**2 + y**2)
    dx = x-150e-6
    return (rho * I * dx) / (4 * np.pi * r**3)

def calculate_activating_function(x: np.ndarray, y: float, I: float, rho: float = 1.0) -> np.ndarray:
    r = np.sqrt((x-150e-6)**2 + y**2)
    dx = x-150e-6
    return (rho * I) / (4 * np.pi) * (2*dx**2 - r**2) / r**5

def plot_field(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, filename: str, I: float = 1e-3):
    plt.figure(figsize=(10, 6))
    plt.plot(x*1e6, y)
    plt.grid(True)
    plt.xlabel('x (µm)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'{title} (I = {I*1e3:.1f} mA)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13)
    
    plt.gca().yaxis.get_offset_text().set_fontsize(13)
    
    plt.tight_layout()
    
    os.makedirs('out', exist_ok=True)
    plt.savefig(os.path.join('out', filename))
    plt.close()

def task_1_1():
    x = np.arange(0, 300.1e-6, 0.1e-6)
    y = 10e-6

    
    currents = [1e-3, -1e-3]  # ±1 mA
    
    for I in currents:
        potential = calculate_potential(x, y, I)
        efield = calculate_efield(x, y, I)
        act_func = calculate_activating_function(x, y, I)
        
        current_str = 'pos' if I > 0 else 'neg'
        
        plot_field(x, potential, 'Extracellular Potential', 'V (V)', 
                  f'potential_{current_str}.png', I)
        plot_field(x, efield, 'Electric Field', 'E (V/m)', 
                  f'efield_{current_str}.png', I)
        plot_field(x, act_func, 'Activating Function', 'A (V/m²)', 
                  f'actfunc_{current_str}.png', I)

if __name__ == "__main__":
    task_1_1()

import numpy as np
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt

type Function = Callable[[float, float], float]
type Solver = Callable[[Function, float, float, float], Tuple[float, float]]


def euler(function: Function, V: float, t: float, step: float) -> Tuple[float, float]:
    Vi = V + step * function(V, t)
    ti = t + step
    return (Vi, ti)


def heun(function: Function, V: float, t: float, step: float) -> Tuple[float, float]:
    V1 = function(V, t)
    V2 = function(V + step * V1, t + step)
    Vi = V + (step / 2) * (V1 + V2)
    ti = t + step
    return (Vi, ti)


def exponential_euler(function: Function, V: float, t: float, step: float) -> Tuple[float, float]:
    Vi = V * np.exp(-step) + step * np.exp(-step) * function(V, t)
    ti = t + step
    return (Vi, ti)


def approx(solver: Solver, function: Function, V0: float, t0: float, ttime: float, step: float) -> List[Tuple[float, float]]:
    t = t0
    V = V0
    results = [(t, V)]
    
    end = t0 + ttime
    while t < end:
        V, t = solver(function, V, t, step)
        results.append((t, V))

    return results


def plotrox(
    solver: Solver,
    title: str,
    function: Function,
    function_name: str,
    V0: float, t0: float, 
    ttime: float, steps: List[float],
    save: bool = False,
):
    plt.figure(figsize=(12, 6))
    for step in steps:
        result = approx(solver, function, V0, t0, ttime, step)
        t_vals, V_vals = zip(*result)
        plt.plot(t_vals, V_vals, label=f"$\\Delta t = {step}s$")
    plt.title(f"{title} Approximation for: {function_name}")
    plt.xlabel("t(s)")
    plt.ylabel("V(V)")
    plt.xticks(np.arange(t0, t0 + ttime + 1, 1))
    plt.yticks(np.arange(-4, 8 + 1, 2))
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"out/{title}.svg")
    else:
        plt.show()
    plt.close()



def ex1(save: bool = False):
    V0 = -3
    t0 = -4
    teatime = 9
    steps = [1.5, 0.75, 0.1]
    function_name = "$\\frac{dV}{dt} = 1 - V -t$"
    def function(V, t) -> float: return 1.0 - V - t

    plotrox(euler, "Euler", function, function_name, V0, t0, teatime, steps, save)  
    plotrox(heun, "Heun", function, function_name, V0, t0, teatime, steps, save)  
    plotrox(exponential_euler, "Exponential Euler", function, function_name, V0, t0, teatime, steps, save)  



def lif_neuron(
    Iamp: float,
    freq: float,
    Cm: float,
    gleak: float,
    Vrest: float,
    Vthr: float,
    Vspike: float,
    T: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(0, T, dt)
    Iinput = Iamp * np.abs(np.sin(2 * np.pi * freq * time))
    V = np.zeros_like(time)
    V[0] = Vrest
    
    for n in range(1, len(time)):
        if V[n - 1] < Vthr:
            V[n] = V[n - 1] + (dt / Cm) * (-gleak * (V[n - 1] - Vrest) + Iinput[n - 1])
        elif Vthr <= V[n - 1] < Vspike:
            V[n] = Vspike
        elif Vspike <= V[n - 1]:
            V[n] = Vrest

    return time, V, Iinput



def plot_lif(T: np.ndarray, V: np.ndarray, Iin: np.ndarray, Iamp: float, save: bool = False):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(T * 1e3, V * 1e3)
    plt.title(f"Membrane Voltage and Input Current ({Iamp} µA)")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(T * 1e3, Iin * 1e6)  # Convert to ms and µA
    plt.xlabel("Time (ms)")
    plt.ylabel("I (µA)")
    plt.grid()

    # plt.tight_layout()
    if save:
        plt.savefig(f"out/lif{int(Iamp)}.svg")
    else:
        plt.show()
    plt.close()

def ex2(save: bool = False):
    Cm = 1e-6
    gleak = 100e-6
    Vrest = -60e-3
    Vthr = -20e-3
    Vspike = 20e-3

    time = 50e-3
    dt = 25e-6
    
    freq = 50

    T10, V10, I10 = lif_neuron(10e-6, freq, Cm, gleak, Vrest, Vthr, Vspike, time, dt)
    T30, V30, I30 = lif_neuron(30e-6, freq, Cm, gleak, Vrest, Vthr, Vspike, time, dt) 
    
    plot_lif(T10, V10, I10, 10, save)
    plot_lif(T30, V30, I30, 30, save)

def main():
    ex1(True)
    ex2(False)


if __name__ == "__main__":
    main()

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def plot1(filename: Optional[str] = None):
    t = np.linspace(-6, 10, 60)
    v = np.linspace(0, -20, 30)

    T, V = np.meshgrid(t, v)
    dT = np.ones(T.shape)
    dV = -10 - V - t
   
    N = np.sqrt(dT**2 + dV**2)
    dTn, dVn = dT/N, dV/N

    t_iso = np.linspace(-6, 10, 100)
    iso_slopes = [-3, -1, 1]
    isos = [-10 - slope - t_iso for slope in iso_slopes] 

    plt.figure(figsize=(12, 6))
    plt.title("Slope Field for $\\frac{dV}{dt} = -10 - V - t$")
    plt.xlabel("t(s)")
    plt.ylabel("V(V)")
    plt.xlim(-6, 10)
    plt.ylim(-20, 0)
    plt.grid(True)
 
    plt.quiver(T, V, dTn, dVn, color="blue", label="slope field", width=0.0015)

    for i, iso in enumerate(isos):
        plt.plot(t_iso, iso, label=f"isocline for {iso_slopes[i]}")

    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot2(filename: Optional[str] = None):
    t = np.linspace(0, 10, 60)
    v = np.linspace(32, 48, 30)

    T, V = np.meshgrid(t, v)
    dT = np.ones(T.shape)
    dV = np.cos(t) - (1/2 * V) + 20 
   
    N = np.sqrt(dT**2 + dV**2)
    dTn, dVn = dT/N, dV/N

    t_iso = np.linspace(0, 10, 100)
    iso_slopes = [-2, 0, 2]
    isos = [2 * (np.cos(t_iso) - slope + 20) for slope in iso_slopes] 

    plt.figure(figsize=(12, 6))
    plt.title("Slope Field for $\\frac{dV}{dt} = cos(t) - V/2 +20$")
    plt.xlabel("t(s)")
    plt.ylabel("V(V)")
    plt.xlim(0, 10)
    plt.ylim(32, 48)
    plt.grid(True)
 
    plt.quiver(T, V, dTn, dVn, color="blue", label="slope field", width=0.0015)

    for i, iso in enumerate(isos):
        plt.plot(t_iso, iso, label=f"isocline for {iso_slopes[i]}")

    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plotEx2(R: float, C: float, Imax: float, D = 0, filename: Optional[str] = None):
    t = np.linspace(0, 6, 50)
    v = np.linspace(20, -20, 30)

    T, V = np.meshgrid(t, v)
    dT = np.ones(T.shape)

    dV = (Imax / C) * np.sin(t) + D/C - (V / (R * C))
   
    N = np.sqrt(dT**2 + dV**2)
    dTn, dVn = dT/N, dV/N

    plt.figure(figsize=(12, 6))
    plt.title("Slope Field for $\\frac{{dV}}{{dt}} = \\frac{I_{max}}{C} \\sin(t) + \\frac{D}{C} - \\frac{V}{RC}$")
    plt.xlabel("t(s)")
    plt.ylabel("V(V)")
    plt.xlim(0, 6)
    plt.ylim(-20, 20)
    plt.grid(True)
 
    plt.quiver(T, V, dTn, dVn, color="blue", label="slope field", width=0.0015)

    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def main() -> None:
    plot1(filename="out/ex1a.svg")
    plot2(filename="out/ex1b.svg")
    plotEx2(1.0, 2.0, 0.0, filename="out/ex2a1.svg")
    plotEx2(1.0, 2.0, 10.0, filename="out/ex2a2.svg")
    plotEx2(1.0, 2.0, 0.0, D=5, filename="out/ex2b1.svg")
    plotEx2(1.0, 2.0, 10.0, D=5, filename="out/ex2b2.svg")

if __name__ == "__main__":
    main()


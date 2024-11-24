from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_signal(
    frequencies: List[float],
    amplitudes: List[float],
    offset: float,
    duration: float, # seconds
    sample_rate: float, # Hz 
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.zeros_like(t) + offset

    for F, A in zip(frequencies, amplitudes):
        # a DC component is just a "constant" y-offset
        if F == 0:
            signal += A
        else:
            signal += A * np.sin(2 * np.pi * F * t)

    return t, signal


def calculate_spectra(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    spectra = np.abs(fft_result) * 2 / n
   
    # From the docs of rfft
    # When `A = rfft(a)` and fs is the sampling frequency, `A[0]` contains
    # the zero-frequency term 0\*fs, which is real due to Hermitian symmetry.

    # If `n` is even, `A[-1]` contains the term representing both positive
    # and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    # real. If `n` is odd, there is no term at fs/2; `A[-1]` contains
    # the largest positive frequency (fs/2\*(n-1)/n), and is complex in the
    # general case.
    
    # => we need to correct these cases!
    # we only have a positive spectrum thus we have to halve a DC as they have no negative one.
    spectra[0] /= 2
    if n % 2 == 0:
        spectra[-1] /= 2
    return freqs, spectra


sample_rate = 12.e3
duration = 1.


def ex1_1(plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    frequencies = [50., 500., 5000.]
    amplitudes = [2., 4., 2.]
    time, signal = generate_signal(frequencies, amplitudes, 0, duration, sample_rate) 

    plt.figure(figsize=(12, 4))
    plt.plot(time * 1000, signal, label="Signal")
    plt.xlim(0, 50)
    plt.xlabel("Time (ms)")
    plt.title("Signal with 50, 500, and 5000Hz")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", ls="--")
    if plot:
        plt.show()
    plt.savefig("out/ex1_1.png")

    return time, signal


def ex2_1(plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    _, signal = ex1_1()

    freq, spectra = calculate_spectra(signal, sample_rate) 
   
    plt.figure(figsize=(12, 4))
    plt.semilogx(freq / 1000, spectra)
    plt.scatter(freq / 1000, spectra, color='blue')
    plt.title("Amplitude Spectrum of the Signal")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", ls="--")
    if plot:
        plt.show()
    plt.savefig("out/ex2_1.png")

    return freq, spectra


def ex1_2(plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    frequencies = [0., 1000., 10000.]
    amplitudes = [3., 5., 3.]
    offset = 0
    time, signal = generate_signal(frequencies, amplitudes, offset, duration, sample_rate) 

    plt.figure(figsize=(12, 4))
    plt.plot(time * 1000, signal, label="Signal")
    plt.xlim(0, 50)
    plt.xlabel("Time (ms)")
    plt.title("Signal with 0, 1000, and 10000Hz")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", ls="--")
    if plot:
        plt.show()
    plt.savefig("out/ex1_2.png")

    return time, signal


def ex2_2(plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    _, signal = ex1_2()

    freq, spectra = calculate_spectra(signal, sample_rate) 
    plt.figure(figsize=(12, 4))
    plt.plot(freq / 1000, spectra)
    plt.scatter(freq / 1000, spectra, color='blue')
    plt.title("Amplitude Spectrum of the Signal")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", ls="--")
    if plot:
        plt.show()
    plt.savefig("out/ex2_2.png")

    return freq, spectra


def main():
    show = False
    # ex1_1(show)
    ex2_1(show)
    # ex1_2(show)
    ex2_2(show)

if __name__ == "__main__":
    main()

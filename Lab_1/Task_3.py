import matplotlib.pyplot as plt
import numpy as np

A = 5 #amplitude
Vm = 5 #modulation frequency
m = 10 #modulation amplitude
DC = 1 #DC offset
Vc = 100 #carrier frequency
k = 1 #frequency index

st = 0 #start time
et = 1 #end time
N = 1000 #number of samples
n = np.linspace(st, et, N)

def sin(a, v, t):
    sin = a * np.sin(2 * np.pi * v * t) #Sine Wave Function
    return sin

def amplitude_modulation(m, Vm, A, Vc, n):
    """
    Function to generate an amplitude
    modulated signal based on a carrier signal
    and modulation signal
    """

    modulation = sin(m, Vm, n)
    carrier = sin(A, Vc, n)
    am = (DC + modulation) * carrier
    return am

def DFT(x): #Discrete Fourier Transform
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = e.dot(x)
    
    return X

def produce_plots(m, Vm, A, Vc, n):
    # Calculate frequency axis
    N = len(n)
    freq = np.fft.fftfreq(N, (n[1] - n[0]))
    
    # Calculate DFT and FFT
    dft_result = DFT(amplitude_modulation(m, Vm, A, Vc, n))
    fft_result = np.fft.fft(amplitude_modulation(m, Vm, A, Vc, n))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # DFT Plot
    plt.subplot(2, 1, 1)
    plt.title("Discrete Fourier Transform")
    plt.plot(freq[:N//2], 2/N * np.abs(dft_result[:N//2]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # FFT Plot
    plt.subplot(2, 1, 2)
    plt.title("Numpy Fast Fourier Transform")
    plt.plot(freq[:N//2], 2/N * np.abs(fft_result[:N//2]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def produce_multiple_dft_plots(param_name, values, m, Vm, A, Vc, n):
    """
    Produce multiple DFT plots varying one parameter.
    param_name: 'm', 'Vm', 'A', or 'Vc'
    values: iterable of values to use for that parameter
    Other parameters default to the module-level values.
    """
    N = len(n)
    freq = np.fft.fftfreq(N, (n[1] - n[0]))[:N//2]
    values = list(values)
    rows = len(values)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3*rows), sharex=True)
    if rows == 1:
        axes = [axes]
    for ax, val in zip(axes, values):
        # select parameters
        params = {'m': m, 'Vm': Vm, 'A': A, 'Vc': Vc}
        params[param_name] = val
        am = amplitude_modulation(params['m'], params['Vm'], params['A'], params['Vc'], n)
        X = DFT(am)
        ax.plot(freq, 2/N * np.abs(X[:N//2]))
        ax.set_title(f"DFT magnitude — {param_name} = {val}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# Example: vary carrier frequency
#produce_multiple_dft_plots('Vc', [80, 90, 100, 110])
produce_plots(m, Vm, A, Vc, n)
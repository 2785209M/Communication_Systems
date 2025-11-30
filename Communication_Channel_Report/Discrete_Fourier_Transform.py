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
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # DFT Plot
    plt.title("Discrete Fourier Transform")
    plt.plot(freq[:N//2], 2/N * np.abs(dft_result[:N//2]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("DFT_Plot")

produce_plots(m, Vm, A, Vc, n)
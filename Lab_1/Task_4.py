import numpy as np
import matplotlib.pyplot as plt

# Signal parameters (from Task 3)
A = 5  # amplitude
Vm = 5  # modulation frequency
m = 10  # modulation amplitude
DC = 1  # DC offset
Vc = 100  # carrier frequency
k = 1  # frequency index

st = 0  # start time
et = 1  # end time
N = 1000  # number of samples
n = np.linspace(st, et, N)

def sin(a, v, t):
    return a * np.sin(2 * np.pi * v * t)

def amplitude_modulation(m, Vm, A, Vc, n):
    modulation = sin(m, Vm, n)
    carrier = sin(A, Vc, n)
    am = (DC + modulation) * carrier
    return am

def bandpass_filter(signal, fs, lowcut, highcut):
    N = len(signal)
    # Get frequency components
    fft_signal = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, 1/fs)
    
    # Create bandpass mask
    mask = np.zeros(N)
    mask[np.abs(freq) >= lowcut] = 1
    mask[np.abs(freq) >= highcut] = 0
    
    # Apply filter
    filtered_signal = np.fft.ifft(fft_signal * mask)
    return filtered_signal.real

# Generate signal
am_signal = amplitude_modulation(m, Vm, A, Vc, n)

# Apply bandpass filter
# Filter frequencies between carrier-modulation and carrier+modulation
lowcut = Vc - Vm
highcut = Vc + Vm
filtered_signal = bandpass_filter(am_signal, N, lowcut, highcut)

# Plot original and filtered signals
def plot():
    # Define different bandwidths
    bandwidths = [
        {'lowcut': Vc - Vm, 'highcut': Vc + Vm, 'label': 'Normal'},
        {'lowcut': Vc - 2*Vm, 'highcut': Vc + 2*Vm, 'label': 'Wide'},
        {'lowcut': Vc - 0.5*Vm, 'highcut': Vc + 0.5*Vm, 'label': 'Narrow'}
    ]
    
    plt.figure(figsize=(12, 12))
    
    # Plot original signal
    plt.subplot(4, 1, 1)
    plt.title("Original AM Signal")
    plt.plot(n, am_signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot filtered signals with different bandwidths
    for i, bw in enumerate(bandwidths, start=2):
        filtered = bandpass_filter(am_signal, fs, bw['lowcut'], bw['highcut'])
        plt.subplot(4, 1, i)
        plt.title(f"Bandpass Filtered Signal ({bw['label']} Bandwidth)")
        plt.plot(n, filtered)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot()
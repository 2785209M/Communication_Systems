import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# --- 1. Define Sampling and Signal Parameters ---
fs = 1000       # Sampling frequency in Hz (1000 samples/sec)
T = 1.0         # Total time duration in seconds
N = int(fs * T) # Total number of samples
f0 = 10         # Fundamental frequency of the signals in Hz

# Create the discrete-time sample index n = 0, 1, ..., N-1
n = np.arange(N)

# --- 2. Generate Time-Domain Signals ---

# 1. Sine Wave
sine_wave = np.sin(2 * np.pi * f0 * n / fs)

# 2. Square Wave (50% duty cycle)
# scipy.signal.square generates a square wave with a 50% duty cycle by default
square_50 = signal.square(2 * np.pi * f0 * n / fs, duty=0.5)

# 3. Square Wave (75% duty cycle)
square_75 = signal.square(2 * np.pi * f0 * n / fs, duty=0.75)

# --- 3. Compute Frequency-Domain Signals (FFT) ---

def get_fft_spectrum(signal, fs, N):
    """Helper function to compute the single-sided FFT spectrum."""
    # Compute the FFT
    fft_raw = np.fft.fft(signal)
    
    # Get the corresponding frequencies for each FFT bin
    # d=1/fs is the sample spacing (time step)
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    # Calculate the normalized magnitude
    # We divide by N to get the correct amplitude scaling
    magnitude = np.abs(fft_raw) / N
    
    # We only need the positive frequency half (from 0 Hz to Nyquist frequency fs/2)
    # The negative half is just a mirror image for real signals
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]
    magnitude_positive = magnitude[positive_mask]
    
    # For a single-sided spectrum, we double the magnitude of all
    # components except the DC (0 Hz) component.
    magnitude_positive[1:] = magnitude_positive[1:] * 2
    
    return freqs_positive, magnitude_positive

# Get spectra for all three signals
freq_sine, mag_sine = get_fft_spectrum(sine_wave, fs, N)
freq_sq50, mag_sq50 = get_fft_spectrum(square_50, fs, N)
freq_sq75, mag_sq75 = get_fft_spectrum(square_75, fs, N)

# --- 4. Plotting ---

fig, ax = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Time Domain vs. Frequency Domain (Discrete Sampling)', fontsize=16, y=1.02)

# Set a threshold for plotting in the frequency domain to hide numerical noise
plot_threshold = 0.01

# --- Plot 1: Sine Wave ---
ax[0, 0].plot(t, sine_wave)
ax[0, 0].set_title(f'Time Domain: Sine Wave ({f0} Hz)')
ax[0, 0].set_xlabel('Time (s)')
ax[0, 0].set_ylabel('Amplitude')
ax[0, 0].set_xlim(0, 0.3) # Show first 3 cycles
ax[0, 0].grid(True)

# Use stem plot for discrete frequency components
mag_sine_plot = np.where(mag_sine > plot_threshold, mag_sine, 0)
ax[0, 1].stem(freq_sine, mag_sine_plot, basefmt=" ")
ax[0, 1].set_title('Frequency Domain: Sine Wave')
ax[0, 1].set_xlabel('Frequency (Hz)')
ax[0, 1].set_ylabel('Normalized Magnitude')
ax[0, 1].set_xlim(0, 100) # Show 0-100 Hz
ax[0, 1].set_ylim(bottom=0)
ax[0, 1].grid(True)

# --- Plot 2: Square Wave (50% Duty) ---
ax[1, 0].plot(t, square_50)
ax[1, 0].set_title('Time Domain: Square Wave (50% Duty Cycle)')
ax[1, 0].set_xlabel('Time (s)')
ax[1, 0].set_ylabel('Amplitude')
ax[1, 0].set_xlim(0, 0.3)
ax[1, 0].grid(True)

mag_sq50_plot = np.where(mag_sq50 > plot_threshold, mag_sq50, 0)
ax[1, 1].stem(freq_sq50, mag_sq50_plot, basefmt=" ")
ax[1, 1].set_title('Frequency Domain: Square Wave (50% Duty Cycle)')
ax[1, 1].set_xlabel('Frequency (Hz)')
ax[1, 1].set_ylabel('Normalized Magnitude')
ax[1, 1].set_xlim(0, 100)
ax[1, 1].set_ylim(bottom=0)
ax[1, 1].grid(True)

# --- Plot 3: Square Wave (75% Duty) ---
ax[2, 0].plot(t, square_75)
ax[2, 0].set_title('Time Domain: Square Wave (75% Duty Cycle)')
ax[2, 0].set_xlabel('Time (s)')
ax[2, 0].set_ylabel('Amplitude')
ax[2, 0].set_xlim(0, 0.3)
ax[2, 0].grid(True)

mag_sq75_plot = np.where(mag_sq75 > plot_threshold, mag_sq75, 0)
ax[2, 1].stem(freq_sq75, mag_sq75_plot, basefmt=" ")
ax[2, 1].set_title('Frequency Domain: Square Wave (75% Duty Cycle)')
ax[2, 1].set_xlabel('Frequency (Hz)')
ax[2, 1].set_ylabel('Normalized Magnitude')
ax[2, 1].set_xlim(0, 100)
ax[2, 1].set_ylim(bottom=0)
ax[2, 1].grid(True)

plt.tight_layout(pad=1.5)
plt.savefig('waveforms_and_spectra.png')
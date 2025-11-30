import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Load the CSV file ---
file_path = "scope_17.csv"
df = pd.read_csv(file_path)

# Rename columns for simplicity
df.columns = ['col1', 'col2']

# --- Remove non-numeric rows (like 'Volt', 'Hertz', etc.) ---
def is_numeric(s):
    try:
        float(str(s).replace('+', '').replace('E', 'e'))
        return True
    except ValueError:
        return False

df = df[df['col1'].apply(is_numeric) & df['col2'].apply(is_numeric)]

# --- Convert to float ---
df = df.apply(lambda x: x.astype(str).str.replace('+', '', regex=False))
df = df.astype(float)
df.columns = ['second', 'Volt']

# --- Extract data ---
t = df['second'].values
v = df['Volt'].values

# --- Compute FFT ---
dt = np.mean(np.diff(t))
fs = 1 / dt  # Sampling frequency
n = len(v)

fft_vals = np.fft.fft(v)
freqs = np.fft.fftfreq(n, d=dt)

# Keep only positive frequencies
mask = freqs > 0
freqs = freqs[mask]
fft_magnitude = np.abs(fft_vals[mask])

# --- Find peaks above 10% of the maximum magnitude ---
peaks, props = find_peaks(fft_magnitude, height=np.max(fft_magnitude) * 0.1)

# --- Print results ---
print("\nDetected Fourier Peaks:")
for f, m in zip(freqs[peaks], fft_magnitude[peaks]):
    print(f"  Frequency: {f:.3f} Hz, Magnitude: {m:.5f}")

# --- Plot FFT Spectrum ---
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_magnitude, label="FFT Magnitude")
plt.plot(freqs[peaks], fft_magnitude[peaks], "ro", label="Peaks")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Spectrum with Peaks")
plt.legend()
plt.tight_layout()

# --- Save instead of showing ---
plt.savefig("fft_peaks.png", dpi=300)
print("\nPlot saved as 'fft_peaks.png' in the current folder.")

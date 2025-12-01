import numpy as np
import matplotlib.pyplot as plt
import os

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()

    data = s.encode("utf-8")  # convert to real bytes

    b = ''.join(format(byte, '08b') for byte in data)
    vector = [int(bit) for bit in b]

    return vector


def modulate_bask(vector, samples_per_bit=1000, freq=1):
    """
    Perform Binary Amplitude Shift Keying (BASK / OOK) modulation.
    
    vector: list of 0/1 integers
    samples_per_bit: samples representing each bit
    freq: carrier frequency in Hz
    """
    t = np.linspace(0, 1, samples_per_bit)          # time for one bit
    carrier = np.sin(2 * np.pi * freq * t)          # one-bit carrier

    full_carrier = np.tile(carrier, len(vector))    # repeat carrier for each bit
    modulation_signal = np.repeat(vector, samples_per_bit)

    modulated = full_carrier * modulation_signal

    # Build full time axis for entire signal
    total_time = np.linspace(0, len(vector), len(modulated))

    return total_time, modulated

def demodulate_bask(modulated_signal, samples_per_bit, threshold=0.4):
    # Threshold => threshold separating rounding for 0 and 1
    num_bits = len(modulated_signal) // samples_per_bit
    recovered = []

    for i in range(num_bits):
        block = modulated_signal[i*samples_per_bit : (i+1)*samples_per_bit]
        amplitude = np.mean(np.abs(block))           # envelope detection
        recovered.append(1 if amplitude > threshold else 0)

    return recovered

def bits_to_text(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Binary length is not a multiple of 8")

    # group into bytes
    byte_values = [
        int(''.join(str(bit) for bit in bits[i:i+8]), 2)
        for i in range(0, len(bits), 8)
    ]

    data = bytes(byte_values)
    return data.decode("utf-8")


def plot_bits(time, modulated_signal, samples_per_bit, bits=5):
    # Plot the first N bits of the modulated signal.
    N = bits * samples_per_bit

    plt.figure(figsize=(10,4))
    plt.plot(time[:N], modulated_signal[:N])
    plt.title(f"First {bits} BASK-modulated bits")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

vector = read_text("/home/james/University/Communication_Systems/Communication_Channel_Report/Input.txt")

time, modulated = modulate_bask(vector, samples_per_bit=1000, freq=5)
recovered = demodulate_bask(modulated, samples_per_bit=1000)
print("Original bits:", len(vector))
print("Recovered bits:", len(recovered))
print("Difference:", len(recovered) - len(vector))
print("Original first 40 bits:", vector[:40])
print("Recovered first 40 bits:", recovered[:40])
errors = sum(1 for a,b in zip(vector, recovered) if a!=b)
print("Total bit errors:", errors)

recovered_text = bits_to_text(recovered)

print("Recovered text:", recovered_text)
plot_bits(time, modulated, samples_per_bit=1000, bits=20)
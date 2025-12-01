import matplotlib.pyplot as plt
import numpy as np

# Read the text file
f = open("/home/james/University/Communication_Systems/Communication_Channel_Report/Input.txt")
s = f.read()

b = ''.join(format(ord(char), '08b') for char in s)

vector = [int(bit) for bit in b]

# Modulate a sin wave to carry the signal
samples_per_bit = 1000
freq = 5
t = np.linspace(0, 1, samples_per_bit)
bits = 20
N = bits * samples_per_bit

carrier_signal = np.sin(2*np.pi*freq*t)

full_carrier = np.tile(carrier_signal, len(vector))

modulation_signal = np.repeat(vector, samples_per_bit)

modulated_signal = full_carrier * modulation_signal

T = np.linspace(0, len(vector), len(modulated_signal))

plt.plot(T[:N], modulated_signal[:N])
plt.show()
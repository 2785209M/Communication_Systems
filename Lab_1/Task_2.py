import matplotlib.pyplot as plt
import numpy as np

A = 5 #amplitude
Vm = 5 #modulation frequency
m = 10 #modulation amplitude
DC = 1 #DC offset
Vc = 100 #carrier frequency

st = 0 #start time
et = 1 #end time
ns = 1000 #number of samples
t = np.linspace(st, et, ns)

def sin(a, v, t):
    sin = a * np.sin(2 * np.pi * v * t) #Sine Wave Function
    return sin

modulation = sin(m, Vm, t)
carrier = sin(A, Vc, t)

def amplitude_modulation(DC, modulation, carrier):
    am = (DC + modulation) * carrier
    return am

plt.subplot(3, 1, 1)
plt.plot(t, carrier)
plt.title(f"carrier signal: frequency = {Vc}")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(t, modulation)
plt.title(f"modulation signal: frequency = {Vm}")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(t, amplitude_modulation(DC, modulation, carrier))
plt.title(f"modulated signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
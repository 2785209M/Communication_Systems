import matplotlib.pyplot as plt
import numpy as np

A = 5 # Amplitude
Vm = 5 # Modulation frequency
m = 10 # Modulation amplitude
DC = 1 # DC offset
Vc = 100 # Carrier frequency

st = 0 # Start time
et = 1 # End time
ns = 1000 # Number of samples
t = np.linspace(st, et, ns)

def sin(A, V, t):
    sin = A * np.sin(2 * np.pi * V * t) #Sine Wave Function
    return sin

def amplitude_modulation(m, Vm, A, Vc, DC, n):
    modulation = sin(m, Vm, n)
    carrier = sin(A, Vc, n)
    am = (DC + modulation) * carrier
    return am
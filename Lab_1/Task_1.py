import matplotlib.pyplot as plt
import numpy as np

a = 5 #amplitude
v = [1, 5, 10, 20, 25] #frequencies

st = 0 #start time
et = 1 #end time
ns = [1000, 500, 250, 125, 75] #number of samples

def sin(a, v, t):
    sin = a * np.sin(2 * np.pi * v * t) #Sine Wave Function
    return sin

#Formatting the graph
for i in range(len(v)):
    t = np.linspace(st, et, ns[i]) #time samples
    plt.subplot(len(v), 1, i+1) #(rows, cols, index)
    plt.plot(t, sin(a, v[i], t)) #plot the sine wave
    plt.title(f"Sine Wave with frequency {v[i]} HZ and {t.size} samples")
    
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
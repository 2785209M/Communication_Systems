import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Modulation as mod

# read the image file
img = cv2.imread('../image.png', 0)
if img is None:
    raise FileNotFoundError("Image not Found. Check Path and Filename.")

# converting to its binary form
ret, bw_img = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)

# Convert the binary image to a vector
vector = bw_img.flatten()
print(vector)

# Modulate a sin wave to carry the signal
samples_per_bit = 50
f = 10
t = np.linspace(0, 1, samples_per_bit)

carrier_signal = np.sin(2*np.pi*f*t)

full_carrier = np.tile(carrier_signal, len(vector))

modulation_signal = np.repeat(vector, samples_per_bit)

modulated_signal = full_carrier * modulation_signal

print(modulated_signal[9000:10000])

# Display Data (scale back to 0-255 for correct visualization)
plt.imshow(bw_img * 255, cmap = "gray")
plt.title("Binary")
plt.axis("off")
plt.show()
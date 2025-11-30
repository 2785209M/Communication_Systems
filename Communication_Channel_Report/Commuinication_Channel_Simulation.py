# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO

# Read the image file
img = cv2.imread("photo.png")
if img is None:
    raise FileNotFoundError("Image not found. Check your path.")

# convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# compute 2D FFT and shift the zero-frequency component to the center
f = np.fft.fft2(img_gray)
fshift = np.fft.fftshift(f)

# compute magnitude spectrum (use a small epsilon to avoid log(0))
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

# Decompressed Image
decompression = np.fft.ifft2(f)
img_decomp = np.abs(decompression)

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_decomp, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.show()
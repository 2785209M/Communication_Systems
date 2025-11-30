import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO

import Discrete_Fourier_Transform as DFT

# Read the image file
img = cv2.imread("photo.png")
if img is None:
    raise FileNotFoundError("Image not found. Check your path.")

# convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
plt.title('Decompressed Image')
plt.axis('off')

plt.show()
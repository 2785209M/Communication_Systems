import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("photo.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
orig_bytes = img.nbytes

# --- FFT ---
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

# keep only X% largest
keep_ratio = 0.05
N = img.size
K = int(N * keep_ratio)

# find threshold
mag = np.abs(F_shift).flatten()
thresh = np.partition(mag, -K)[-K]

# mask of values to keep
mask = np.abs(F_shift) >= thresh

# extract sparse coefficients
coords = np.column_stack(np.where(mask))       # int32
values = F_shift[mask].astype(np.complex64)    # 8 bytes per complex

# compressed memory size
compressed_bytes = coords.nbytes + values.nbytes

# --- Reconstruction ---
F_sparse = np.zeros_like(F_shift)
F_sparse[mask] = values

img_rec = np.fft.ifft2(np.fft.ifftshift(F_sparse))
img_rec = np.abs(img_rec).astype(np.uint8)

# print results
print("\n--- REAL Compression Results ---")
print(f"Original bytes:      {orig_bytes}")
print(f"Sparse coords bytes: {coords.nbytes}")
print(f"Sparse values bytes: {values.nbytes}")
print(f"Compressed total:    {compressed_bytes}")
print(f"Compression ratio:   {orig_bytes / compressed_bytes:.2f}x")
print(f"Kept coefficients:   {K}/{N} ({keep_ratio*100:.1f}%)")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(img_rec, cmap='gray'); plt.title("Reconstruction"); plt.axis("off")
plt.show()
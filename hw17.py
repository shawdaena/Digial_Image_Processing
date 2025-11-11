import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ---------- Load Image ----------
img = cv2.imread('/home/enoconda/Desktop/DIP/Image/HVD.png', cv2.IMREAD_GRAYSCALE)

# Resize for consistency
img = cv2.resize(img, (256, 256))

# ---------- 1. Discrete Cosine Transform (DCT) ----------
img_dct = cv2.dct(np.float32(img))
img_dct_log = np.log(abs(img_dct) + 1)

# ---------- 2. Discrete Wavelet Transform (DWT) ----------
coeffs2 = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs2

# ---------- 3. Discrete Fourier Transform (DFT) ----------
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

# ---------- Visualization ----------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_dct_log, cmap='gray')
plt.title('DCT Spectrum (log)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cA, cmap='gray')
plt.title('DWT Approximation (cA)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cH, cmap='gray')
plt.title('DWT Horizontal Detail (cH)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cV, cmap='gray')
plt.title('DWT Vertical Detail (cV)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('DFT Magnitude Spectrum')
plt.axis('off')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ----------- DFT / IDFT -----------
def dft(img):
    return np.fft.fftshift(np.fft.fft2(img))

def idft(fshift):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))

# ----------- Butterworth Filters -----------
def butterworth_lpf(shape, D0, n=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(u**2 + v**2)
    return 1 / (1 + (D / D0)**(2 * n))

def butterworth_hpf(shape, D0, n=2):
    return 1 - butterworth_lpf(shape, D0, n)

def butterworth_bpf(shape, D_low, D_high, n=2):
    return butterworth_lpf(shape, D_high, n) - butterworth_lpf(shape, D_low, n)c

# ----------- Gaussian Filters -----------
def gaussian_lpf(shape, D0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(u**2 + v**2)
    return np.exp(-(D**2) / (2 * (D0**2)))

def gaussian_hpf(shape, D0):
    return 1 - gaussian_lpf(shape, D0)

def gaussian_bpf(shape, D0, w):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(u**2 + v**2)
    return np.exp(-((D**2 - D0**2)**2) / (D**2 * (w**2) + 1e-8))

# ----------- Apply Filter -----------
def apply_filter(img, mask):
    fshift = dft(img)
    filtered = fshift * mask
    return idft(filtered)

# ----------- Main -----------
def main():
    # -----------Load images-----------
    low = cv2.imread("/home/enoconda/Desktop/DIP/Image/LowLowCon.png", 0)
    normal = cv2.imread("/home/enoconda/Desktop/DIP/Image/tulip.png", 0)
    high = cv2.imread("/home/enoconda/Desktop/DIP/Image/HighHighCon.png", 0)

    images = [('Low Contrast', low),('Normal Contrast', normal),('High Contrast', high)]

    D0 = 40
    D_low = 20
    D_high = 60
    w = 20
    n = 2

    # ----------- Figure 1: Butterworth -----------
    plt.figure(1, figsize=(8, 8))
    plot_index = 1

    for title, img in images:
        shape = img.shape

        blpf = apply_filter(img, butterworth_lpf(shape, D0, n))
        bhpf = apply_filter(img, butterworth_hpf(shape, D0, n))
        bbpf = apply_filter(img, butterworth_bpf(shape, D_low, D_high, n))

        for result, name in [
            (blpf, 'Butterworth LPF'),
            (bhpf, 'Butterworth HPF'),
            (bbpf, 'Butterworth BPF'),
        ]:
            plt.subplot(len(images), 3, plot_index)
            plt.imshow(result, cmap='gray')
            plt.title(f'{title}\n{name}', fontsize=9)
            plt.axis('off')
            plot_index += 1

    plt.tight_layout()
    plt.suptitle("Figure 1: Butterworth Filters", fontsize=14, y=1.02)
    plt.show()

    # ----------- Figure 2: Gaussian -----------
    plt.figure(2, figsize=(8, 8))
    plot_index = 1

    for title, img in images:
        shape = img.shape

        glpf = apply_filter(img, gaussian_lpf(shape, D0))
        ghpf = apply_filter(img, gaussian_hpf(shape, D0))
        gbpf = apply_filter(img, gaussian_bpf(shape, D0, w))

        for result, name in [
            (glpf, 'Gaussian LPF'),
            (ghpf, 'Gaussian HPF'),
            (gbpf, 'Gaussian BPF'),
        ]:
            plt.subplot(len(images), 3, plot_index)
            plt.imshow(result, cmap='gray')
            plt.title(f'{title}\n{name}', fontsize=9)
            plt.axis('off')
            plot_index += 1

    plt.tight_layout()
    plt.suptitle("Figure 2: Gaussian Filters", fontsize=14, y=1.02)
    plt.show()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------- DFT function -----------
def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

# ----------- Low-pass filtering -----------
def low_pass_filtering(dft_shift, radius):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    fshift = dft_shift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)
    return img_back

# ----------- High-pass filtering -----------
def high_pass_filtering(dft_shift, radius):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)
    fshift = dft_shift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)
    return img_back

# ----------- Band-pass filtering -----------
def band_pass_filtering(dft_shift, r_low, r_high):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2
    low_mask = np.zeros((rows, cols), np.uint8)
    high_mask = np.ones((rows, cols), np.uint8)
    cv2.circle(low_mask, (ccol, crow), r_high, 1, -1)
    cv2.circle(high_mask, (ccol, crow), r_low, 0, -1)
    mask = low_mask * high_mask
    fshift = dft_shift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)
    return img_back

# ----------- Main -----------
def main():
    # Paths
    low_path = '/home/enoconda/Desktop/DIP/Image/LowLowCon.png'
    medium_low_path = '/home/enoconda/Desktop/DIP/Image/lowContrast.png'
    normal_path = '/home/enoconda/Desktop/DIP/Image/tulip.png'
    medium_high_path = '/home/enoconda/Desktop/DIP/Image/HighContrast.png'
    high_path = '/home/enoconda/Desktop/DIP/Image/HighHighCon.png'


    # Read images in grayscale
    low = cv2.imread(low_path, cv2.IMREAD_GRAYSCALE)
    medium_low = cv2.imread(medium_low_path, cv2.IMREAD_GRAYSCALE)
    normal = cv2.imread(normal_path, cv2.IMREAD_GRAYSCALE)
    medium_high = cv2.imread(medium_high_path, cv2.IMREAD_GRAYSCALE)
    high = cv2.imread(high_path, cv2.IMREAD_GRAYSCALE)

    # Collect all images
    images = [('Low Contrast', low), ('Medium Low Contrast', medium_low),('Normal', normal), ('Medium High Contrast', medium_high), ('High Contrast', high)]

    # Prepare figure
    plt.figure(figsize=(8, 8))

    plot_index = 1
    for title, img in images:
        dft_shift = dft(img)
        r_low = img.shape[0] // 6
        r_high = img.shape[0] // 4

        # Low-pass
        low_pass = low_pass_filtering(dft_shift, r_high)
        plt.subplot(5, 3, plot_index)
        plt.imshow(low_pass, cmap='gray')
        plt.title(f'{title} - Low-pass')
        plt.axis('off')
        plot_index += 1

        # High-pass
        high_pass = high_pass_filtering(dft_shift, r_low)
        plt.subplot(5, 3, plot_index)
        plt.imshow(high_pass, cmap='gray')
        plt.title(f'{title} - High-pass')
        plt.axis('off')
        plot_index += 1

        # Band-pass
        band_pass = band_pass_filtering(dft_shift, r_low, r_high)
        plt.subplot(5, 3, plot_index)
        plt.imshow(band_pass, cmap='gray')
        plt.title(f'{title} - Band-pass')
        plt.axis('off')
        plot_index += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
